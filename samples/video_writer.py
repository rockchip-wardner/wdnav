import cv2
import wdnav


def test_h264_encode():
    frame = cv2.imread('/userdata/test/demo.jpg')
    writer = wdnav.VideoWriter('/tmp/test.264')  # 不写的参数话默认尺寸为frame的尺寸
    for i in range(100):
        writer.write(frame)  # 这里才会初始化编码器
    writer.release()
    print('h264 raw done: backend =', writer.backend)


def test_hevc_encode():
    frame = cv2.imread('/userdata/test/demo.jpg')
    writer = wdnav.VideoWriter(
        '/tmp/test.265',
        fps=30,
        frame_size=(1920, 1080),  # 分辨率手动设置后编码前会经过RGA进行缩放
        codec=wdnav.VideoWriter.CODEC_HEVC_RAW,
        gop=60,
        bitrate=2_000_000,
        rc_mode='cbr',
    )
    for i in range(100):
        writer.write(frame)
    writer.release()
    print('hevc raw done: backend =', writer.backend)


def test_mp4_encode():
    frame = cv2.imread('/userdata/test/demo.jpg')
    h, w = frame.shape[:2]
    writer = wdnav.VideoWriter(
        '/tmp/test.mp4',
        fps=30,
        frame_size=(w, h),
        codec='h264',
        profile='baseline',
        gop=60,
        bitrate=2_000_000,
        rc_mode='cbr',
    )
    for i in range(100):
        writer.write(frame)
    writer.release()
    print('mp4 done: backend =', writer.backend)


def test_mkv_encode():
    frame = cv2.imread('/userdata/test/demo.jpg')
    h, w = frame.shape[:2]
    writer = wdnav.VideoWriter(
        '/tmp/test.mkv',
        fps=30,
        frame_size=(w, h),
        codec='h264',
        profile='baseline',
        gop=60,
        bitrate=2_000_000,
        rc_mode='cbr',
    )
    for i in range(100):
        writer.write(frame)
    writer.release()
    print('mkv done: backend =', writer.backend)


def test_jpeg_encode():
    frame = cv2.imread('/userdata/test/demo.jpg')
    h, w = frame.shape[:2]
    writer = wdnav.VideoWriter(
        '/tmp/test.jpg',
        fps=30,
        frame_size=(w, h),
        codec=wdnav.VideoWriter.CODEC_JPEG,
        q_factor=80,
    )
    writer.write(frame)
    writer.release()
    print('jpeg done: backend =', writer.backend)


if __name__ == "__main__":
    test_h264_encode()
    test_hevc_encode()
    test_mp4_encode()
    test_mkv_encode()
    test_jpeg_encode()
