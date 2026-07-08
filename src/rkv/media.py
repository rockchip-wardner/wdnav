"""Media Controller (``/dev/mediaX``) topology inspection and configuration.

The Linux Media Controller API lets you describe a camera pipeline as a
graph of *entities* (sensors, ISPs, capture nodes) connected by *links*.
On Rockchip SOCs the typical pipeline looks like::

    sensor -> mipi-csi2 -> rkcif -> /dev/video0
                                 -> rkisp -> /dev/video11

Each video node (``/dev/videoX``) is just one endpoint of this graph —
opening one for capture requires the graph to be wired first. That's
what this module does:

- :class:`MediaDevice` — wraps the ``media-ctl`` CLI for a single
  ``/dev/mediaX`` device, so you can enumerate entities / links,
  enable / disable links, and set pad formats.
- :class:`MediaEntity` / :class:`MediaLink` / :class:`MediaPad` —
  dataclasses returned by the inspection API.

Once the pipeline is configured, hand the resulting capture node
(``/dev/videoX``) to :class:`rkv.video.VideoCapture` to actually pull
frames.

Typical usage on a Rockchip board::

    import rkv
    from rkv.media import MediaDevice

    dev = MediaDevice("/dev/media0")
    for ent in dev.entities():
        print(ent.id, ent.name, ent.device_path)

    # Enable sensor -> CSI link (idempotent; media-ctl deduplicates).
    dev.setup_link("imx415 2-001a":1, "rockchip-csi2-dphy0":0)

    # Set ISP pad format.
    dev.setup_format("rockchip-csi2-dphy0":1, fmt="SRGGB10", width=1920, height=1080)

    cap = rkv.VideoCapture(dev.entities()[0].device_path)
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


# Pad flags (mirrors <linux/media.h> MEDIA_PAD_FL_*).
PAD_FL_SINK = 1
PAD_FL_SOURCE = 2

# Link flags (mirrors <linux/media.h> MEDIA_LNK_FL_*).
LNK_FL_ENABLED = 1
LNK_FL_IMMUTABLE = 2


@dataclass
class MediaPad:
    """A single pad on an entity."""

    index: int
    flags: int                       # bitwise OR of PAD_FL_*


@dataclass
class MediaEntity:
    """A node in the media graph (sensor, subdev, capture, etc.)."""

    id: int
    name: str
    type: str = ""                   # e.g. "Node subtype V4L"
    flags: int = 0
    device_path: Optional[str] = None  # e.g. "/dev/video0"
    pads: List[MediaPad] = field(default_factory=list)


@dataclass
class MediaLink:
    """A directed edge between two pads.

    ``source_entity`` / ``source_pad`` points at the producing pad,
    ``sink_entity`` / ``sink_pad`` at the consuming one. ``enabled`` is
    True when the link is currently active; ``flags`` carries the raw
    kernel flags (immutable, dynamic, etc.).
    """

    source_entity: str
    source_pad: int
    sink_entity: str
    sink_pad: int
    enabled: bool
    flags: int = 0


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


_ENTITY_RE = re.compile(
    r"^\s*-\s+entity\s+(?P<id>\d+):\s+(?P<name>\S+)\s+\((?P<numpads>\d+)\s+pad(?:s)?,\s+(?P<numlinks>\d+)\s+link(?:s)?\)"
)
_TYPE_RE = re.compile(
    r"^\s+type\s+(?P<type>.+?)\s+flags\s+(?P<flags>\d+)"
)
_DEVNODE_RE = re.compile(r"^\s+device node name\s+(?P<path>/\S+)")
_PAD_RE = re.compile(r"^\s*pad(?P<index>\d+):\s+(?P<flags>.+)$")
# Inbound link under a sink pad: `        <- "name":N [FLAGS]`
_INBOUND_RE = re.compile(
    r"^\s+<-\s+\"(?P<name>[^\"]+)\":(?P<pad>\d+)(?:\s+\[(?P<flags>[^\]]*)\])?"
)


def _parse_pad_flags(s: str) -> int:
    flags = 0
    s = s.strip()
    if "Sink" in s:
        flags |= PAD_FL_SINK
    if "Source" in s:
        flags |= PAD_FL_SOURCE
    return flags


def _parse_link_flags(s: str) -> int:
    if not s:
        return 0
    flags = 0
    if "ENABLED" in s:
        flags |= LNK_FL_ENABLED
    if "IMMUTABLE" in s:
        flags |= LNK_FL_IMMUTABLE
    return flags


def _parse_topology(text: str) -> Tuple[List[MediaEntity], List[MediaLink]]:
    """Parse ``media-ctl -p`` output into entities + links.

    The format is line-oriented: an entity block lists its pads, and
    each sink pad lists its inbound links (``<- "src":N [FLAGS]``).
    Outbound links (``->``) aren't printed separately; we reconstruct
    them from the inbound lists. Because two pads can both point at
    each other in odd topologies, we dedupe pairs.
    """
    entities: List[MediaEntity] = []
    links: List[MediaLink] = []
    seen_links: set = set()

    cur: Optional[MediaEntity] = None
    cur_pad_flags: int = 0

    for raw in text.splitlines():
        m = _ENTITY_RE.match(raw)
        if m:
            cur = MediaEntity(
                id=int(m.group("id")),
                name=m.group("name"),
            )
            entities.append(cur)
            cur_pad_flags = 0
            continue
        if cur is None:
            continue  # skip header / driver-info block

        m = _TYPE_RE.match(raw)
        if m:
            cur.type = m.group("type").strip()
            cur.flags = int(m.group("flags"))
            continue

        m = _DEVNODE_RE.match(raw)
        if m:
            cur.device_path = m.group("path")
            continue

        m = _PAD_RE.match(raw)
        if m:
            pad = MediaPad(
                index=int(m.group("index")),
                flags=_parse_pad_flags(m.group("flags")),
            )
            cur.pads.append(pad)
            cur_pad_flags = pad.flags
            continue

        m = _INBOUND_RE.match(raw)
        if m and (cur_pad_flags & PAD_FL_SINK):
            src_name = m.group("name")
            src_pad = int(m.group("pad"))
            link_flags = _parse_link_flags(m.group("flags") or "")
            enabled = bool(link_flags & LNK_FL_ENABLED)
            key = (src_name, src_pad, cur.name, len(cur.pads) - 1)
            if key not in seen_links:
                seen_links.add(key)
                links.append(MediaLink(
                    source_entity=src_name,
                    source_pad=src_pad,
                    sink_entity=cur.name,
                    sink_pad=len(cur.pads) - 1,
                    enabled=enabled,
                    flags=link_flags,
                ))

    return entities, links


# ---------------------------------------------------------------------------
# MediaDevice
# ---------------------------------------------------------------------------


class MediaDevice:
    """Thin Python wrapper around the ``media-ctl`` CLI.

    Parameters
    ----------
    path:
        Path to a media device node (e.g. ``"/dev/media0"``) or a bus
        info substring that ``media-ctl -d`` will resolve (e.g.
        ``"platform:rkcif-mipi-lvds2"``).

    The constructor doesn't shell out — it only validates that
    ``media-ctl`` is on ``$PATH``. Call :meth:`refresh` (or any of the
    other inspection methods) to actually fetch the topology.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._media_ctl = shutil.which("media-ctl")
        if self._media_ctl is None:
            raise FileNotFoundError(
                "media-ctl not found on PATH — install v4l-utils"
            )
        self._entities: Optional[List[MediaEntity]] = None
        self._links: Optional[List[MediaLink]] = None

    # -- introspection -----------------------------------------------------

    @property
    def path(self) -> str:
        return self._path

    def refresh(self) -> None:
        """Re-fetch the topology from the kernel."""
        out = self._run("-p")
        ents, links = _parse_topology(out)
        self._entities = ents
        self._links = links

    def entities(self) -> List[MediaEntity]:
        if self._entities is None:
            self.refresh()
        return list(self._entities or ())

    def links(self) -> List[MediaLink]:
        if self._links is None:
            self.refresh()
        return list(self._links or ())

    def find_entity(self, name: str) -> Optional[MediaEntity]:
        for ent in self.entities():
            if ent.name == name:
                return ent
        return None

    def capture_nodes(self) -> List[MediaEntity]:
        """Convenience: entities that have a /dev/videoX device node."""
        return [e for e in self.entities() if e.device_path]

    # -- configuration -----------------------------------------------------

    def setup_link(
        self,
        source: str,
        sink: str,
        enable: bool = True,
        *,
        source_pad: Optional[int] = None,
        sink_pad: Optional[int] = None,
    ) -> None:
        """Enable or disable a single link.

        Both ``source`` and ``sink`` are entity-name strings (matched
        against the topology). ``source_pad`` / ``sink_pad`` default to
        0; set them explicitly when an entity has more than one pad on
        the relevant side.

        Raises :class:`RuntimeError` if either entity isn't in the
        cached topology (call :meth:`refresh` first if you've changed
        the pipeline externally).
        """
        src = self.find_entity(source)
        if src is None:
            raise RuntimeError(f"entity not found: {source!r}")
        snk = self.find_entity(sink)
        if snk is None:
            raise RuntimeError(f"entity not found: {sink!r}")

        if source_pad is None:
            source_pad = self._default_pad(src, PAD_FL_SOURCE)
        if sink_pad is None:
            sink_pad = self._default_pad(snk, PAD_FL_SINK)

        flag = "1" if enable else "0"
        # media-ctl -l '"src":pad -> "snk":pad [flag]'
        spec = '"{0}":{1}->"{2}":{3}[{4}]'.format(
            src.name, source_pad, snk.name, sink_pad, flag
        )
        self._run("-l", spec)
        # topology may have changed — drop cache so the next call refreshes.
        self._invalidate()

    def setup_format(
        self,
        entity: str,
        pad: int,
        *,
        fmt: str,
        width: int,
        height: int,
    ) -> None:
        """Set the media-bus / V4L2 format on a pad.

        ``fmt`` is a FourCC string or media-bus code (e.g.
        ``"SRGGB10"``, ``"NV12"``, ``"UYVY"``).
        """
        spec = '"{0}":{1}[fmt:{2}/{3}x{4}]'.format(
            entity, pad, fmt, width, height
        )
        self._run("-V", spec)
        self._invalidate()

    def reset(self) -> None:
        """Disable every link on this device.

        Equivalent to ``media-ctl -r``; useful for tearing down a
        pipeline before reconfiguring.
        """
        self._run("-r")
        self._invalidate()

    # -- internals ---------------------------------------------------------

    def _run(self, *args: str) -> str:
        cmd = [self._media_ctl, "-d", self._path, *args]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "media-ctl failed (rc={0}): {1}\nstderr: {2}".format(
                    proc.returncode, " ".join(cmd), proc.stderr.strip()
                )
            )
        return proc.stdout

    def _invalidate(self) -> None:
        self._entities = None
        self._links = None

    @staticmethod
    def _default_pad(entity: MediaEntity, want_flag: int) -> int:
        """Pick the first pad with the desired direction."""
        for p in entity.pads:
            if p.flags & want_flag:
                return p.index
        # Fall back to pad 0 — media-ctl accepts this for subdevs whose
        # single pad doesn't carry a Sink/Source label in this dump.
        return 0


__all__ = [
    "MediaDevice",
    "MediaEntity",
    "MediaLink",
    "MediaPad",
    "PAD_FL_SINK",
    "PAD_FL_SOURCE",
    "LNK_FL_ENABLED",
    "LNK_FL_IMMUTABLE",
]