import cv2
import datetime

# Import our optical flow library (Lukas-Kanade and Horn-Schunck).
import optical_flow as of


def main():

    color_plot = True
    quiver_plot = True

    # Load and process images. (img1 and img2). Using 0 to read image in grayscale mode
    img1 = cv2.imread("video3/fr00100.png", 0) # 0 -> image in grayscale.
    img2 = cv2.imread("video3/fr00101.png", 0)
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    
    # ****************************************************************************
    # ***************************** LUKAS-KANADE *********************************
    # Parameters:
    M = 11
    stride = 1

    # LK_optical_flow_v1
    start_v1 = datetime.datetime.now()
    flow_v1 = of.LK_optical_flow_v1(img1, img2, M, stride)
    flow_v1 = of.remove_noise(flow_v1, 0.2)
    end_v1 = datetime.datetime.now()
    print("Tiempo de ejecución (LK_optical_flow_v1): ", (end_v1 - start_v1).total_seconds())

    # LK_optical_flow_v2
    start_v2 = datetime.datetime.now()
    flow_v2 = of.LK_optical_flow_v2(img1, img2, M, stride)
    flow_v2 = of.remove_noise(flow_v2, 0.2)
    end_v2 = datetime.datetime.now()
    print("Tiempo de ejecución (LK_optical_flow_v2): ", (end_v2 - start_v2).total_seconds())

    
    # ****************************************************************************
    # ******************************** HORN-SCHUNCK ******************************
    # Parameters:
    Nit = 25
    alpha = 0.1
    start_v3 = datetime.datetime.now()
    flow_v3 = of.HS_optical_flow(img1, img2, Nit, alpha, delta=10**-1, convergence=True)
    flow_v3 = of.remove_noise(flow_v3, 0.05)
    end_v3 = datetime.datetime.now()
    print("Tiempo de ejecución (HS_optical_flow): ", (end_v3 - start_v3).total_seconds())

    # ****************************************************************************
    # ************************ PLOTING QUIVER AND COLOR **************************
    of.draw_optical_flow(flow_v1, img2, quiver_plot=quiver_plot, scale=0.2, color_plot=color_plot)
    of.draw_optical_flow(flow_v2, img2, quiver_plot=quiver_plot, scale=0.2, color_plot=color_plot)
    of.draw_optical_flow(flow_v3, img2, quiver_plot=quiver_plot, scale=0.000008, color_plot=color_plot)
    #of.draw_optical_flow(flow_v33, img2, quiver_plot=quiver_plot, scale=0.000008, color_plot=color_plot)


if __name__ == '__main__':
    main()
