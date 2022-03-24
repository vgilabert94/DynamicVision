import cv2
import argparse
import os

# Import our optical flow library (Lukas-Kanade and Horn-Schunck).
import optical_flow as of


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", type=str, help="path to where images resides")
    ap.add_argument("-v", "--video", type=str, help= "path to where video resides")
    ap.add_argument("-s", "--stride", type=int, default=2, help= "size of algorithm stride")
    ap.add_argument("-m", "--windows_size", type=int, default=11, help= "size of windows algorithm")
    return vars(ap.parse_args())


def calculate_optical_flow(path, params, mode=True):

    color_plot = True
    quiver_plot = False
    M = params[1]
    stride = params[0]

    if mode == True and os.path.isdir(path):
        first_frame = True
        for file in os.listdir(path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                if first_frame:
                    img_path = os.path.join(path, file)
                    img1 = cv2.imread(img_path, 0)
                    img1 = img1 / 255.
                    first_frame = False
                else:
                    img_path = os.path.join(path, file)
                    img2 = cv2.imread(img_path, 0)
                    img2 = img2 / 255.
                    flow = of.LK_optical_flow_v2(img1, img2, M, stride)
                    flow = of.remove_noise(flow, 0.2)
                    of.draw_optical_flow(flow, img2, quiver_plot=quiver_plot, scale=0.2, color_plot=color_plot)
                    img1 = img2.copy()

    elif mode == False and os.path.isfile(path):
        cap = cv2.VideoCapture(path)
        frame_ok, img1 = cap.read()
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1 = img1 / 255.
        while (cap.isOpened()):
            frame_ok, img2 = cap.read()
            if frame_ok:
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                img2 = img2 / 255.
                flow = of.LK_optical_flow_v2(img1, img2, M, stride)
                flow = of.remove_noise(flow, 0.2)
                of.draw_optical_flow(flow, img2, quiver_plot=quiver_plot, color_plot=color_plot)
                img1 = img2.copy()
            
    else:
        print("Input error. Check paths.")


if __name__ == '__main__':
    args = parse_arguments()
    path_imgs = args['images']
    path_video = args['video']   
    params = [args['stride'], args['windows_size']]

    if path_imgs is not None:
        # Apply OF to image dataset.
        calculate_optical_flow(path_imgs, params, mode=True)
    else:
        # Apply OF to video dataset.
        calculate_optical_flow(path_video, params, mode=False)
