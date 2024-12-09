import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import time

class RecordVideo:
    def __init__(self, save_pth, save_size=(1920, 1080), save_fps=8):
        self.save_pth = save_pth
        self.save_fps = save_fps
        self.save_size = save_size
        self.run_flag = True
        b_thread = threading.Thread(target=self.record_thread)
        b_thread.daemon = True
        b_thread.start()

    def stop_record(self):
        self.run_flag = False
        time.sleep(3)

    def record_thread(self):
        writer = cv2.VideoWriter(self.save_pth, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.save_fps, self.save_size)
        pipeline = rs.pipeline()
        config = rs.config()
        with open("/home/ubuntu/hand/calculate/0_datasets_allegro_hand/deviceid.txt",'r') as f:
            id=f.read().splitlines()[0]
            config.enable_device(id)
        config.enable_stream(rs.stream.color, self.save_size[0], self.save_size[1], rs.format.bgr8, self.save_fps)
        pipeline.start(config)
        print('Depth Camera init OK!!!')
        writer_flag = True
        try:
            print('Start processing, press k to pause or continue and q to quite')
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                # convert to numpy array
                color_image = np.asanyarray(color_frame.get_data())

                if writer_flag:
                    writer.write(color_image)
                else:
                    paused_warning = 'Capture is paused,  k to continue.....'
                    cv2.putText(color_image, paused_warning, (int(color_image.shape[0]/10), int(color_image.shape[1]/12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
                    print(paused_warning)

                #===show RGB image=====#
                cv2.namedWindow('RGB_video', cv2.WINDOW_NORMAL)
                cv2.imshow('RGB_video', color_image)

                # ====== input waiting====== #
                k = cv2.waitKey(1)
                if k == ord('q') or self.run_flag == False: # input 'q' , close the windows,exit
                    cv2.destroyAllWindows()
                    break

                # pause capturing , and continue
                if k == ord('k') and writer_flag is True:
                    writer_flag = False
                    print('Now is pause')
                elif k == ord('k') and writer_flag is False:
                    writer_flag = True
                    print('continue capturing')

        finally:
            # Stop streaming
            print('the video save to :' + self.save_pth)
            writer.release()
            pipeline.stop()
            print('pipeline Stop, writer release success.')
            exit(0)

if __name__ == '__main__':
    recorder = RecordVideo('a.mp4', save_size=(1920, 1080), save_fps=8)
    time.sleep(5)
    recorder.stop_record()