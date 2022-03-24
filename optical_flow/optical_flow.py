import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_gradients_optical_flow(img1, img2, Show=False):
    kernelX = np.array([[-1, 1], [-1, 1]]) * 0.25
    kernelY = np.array([[-1, -1], [1, 1]]) * 0.25
    kernelT = np.ones([2, 2]) * 0.25

    # fx -> Ix = (Ix1 + Ix2 ) / 2
    Ix = (cv2.filter2D(img1, -1, kernelX) + cv2.filter2D(img2, -1, kernelX)) / 2
    # fy -> Iy = (Iy1 + Iy2 ) / 2
    Iy = (cv2.filter2D(img1, -1, kernelY) + cv2.filter2D(img2, -1, kernelY)) / 2
    # ft -> It = It2 - It1
    It = cv2.filter2D(img2, -1, kernelT) + cv2.filter2D(img1, -1, -kernelT)

    if Show:
        cv2.imshow("Ix", Ix)
        cv2.imshow("Iy", Iy)
        cv2.imshow("It", It)
        cv2.waitKey(0)

    return Ix, Iy, It


def LK_optical_flow_v1(img1, img2, M=3, stride=1):
    if len(img1.shape) < 3 and len(img1.shape) < 3:
        dim = img1.shape
        flow = np.zeros((dim[0], dim[1], 2))

        Ix, Iy, It = get_gradients_optical_flow(img1, img2)

        for u in range(M, dim[0] - M + 1, stride):
            for v in range(M, dim[1] - M + 1, stride):
                sumIx2 = np.sum(np.power(Ix[u:u+M, v:v+M], 2))
                sumIxIy = np.sum(np.multiply(Ix[u:u+M, v:v+M], Iy[u:u+M, v:v+M]))
                sumIy2 = np.sum(np.power(Iy[u:u+M, v:v+M], 2))
                sumIxIt = np.sum(np.multiply(Ix[u:u+M, v:v+M], It[u:u+M, v:v+M]))
                sumIyIt = np.sum(np.multiply(Iy[u:u+M, v:v+M], It[u:u+M, v:v+M]))

                A = np.array([[sumIx2, sumIxIy], [sumIxIy, sumIy2]])
                b = np.array([[-sumIxIt], [-sumIyIt]])
                A_inv = np.linalg.pinv(A)
                result = np.dot(A_inv, b)
                flow[u, v, 0] = result[0]
                flow[u, v, 1] = result[1]

        return flow

    else:
        print("Error input data. Images must be in grayscale.")
        return


def LK_optical_flow_v2(img1, img2, M=3, stride=1):
    if len(img1.shape) < 3 and len(img1.shape) < 3:
        dim = img1.shape
        flow = np.zeros((dim[0], dim[1], 2))

        Ix, Iy, It = get_gradients_optical_flow(img1, img2)

        for u in range(M, dim[0] - M + 1, stride):
            for v in range(M, dim[1] - M + 1, stride):
                sumIx2 = np.sum(np.power(Ix[u:u+M, v:v+M], 2))
                sumIxIy = np.sum(np.multiply(Ix[u:u+M, v:v+M], Iy[u:u+M, v:v+M]))
                sumIy2 = np.sum(np.power(Iy[u:u+M, v:v+M], 2))
                sumIxIt = np.sum(np.multiply(Ix[u:u+M, v:v+M], It[u:u+M, v:v+M]))
                sumIyIt = np.sum(np.multiply(Iy[u:u+M, v:v+M], It[u:u+M, v:v+M]))

                denom = ((sumIx2 * sumIy2) - (sumIxIy * sumIxIy))
                if denom != 0.0:
                    flow[u, v, 0] = ((-sumIy2 * sumIxIt) + (sumIxIy * sumIyIt)) / denom
                    flow[u, v, 1] = ((sumIxIy * sumIxIt) - (sumIx2 * sumIyIt)) / denom
                else:
                    flow[u, v, 0] = 0
                    flow[u, v, 1] = 0
        return flow

    else:
        print("Error input data. Images must be in grayscale.")
        return


def HS_optical_flow(img1, img2, its=300, alpha=2, delta=10**-1, convergence=True):
    if len(img1.shape) < 3 and len(img1.shape) < 3:
        dim = img1.shape
        kernel = np.array([[1/12, 1/6, 1/12], [1/6, 0, 1/6], [1/12, 1/6, 1/12]])
        U = np.zeros_like(img1)
        V = np.zeros_like(img1)
        flow = np.zeros((dim[0], dim[1], 2))

        Ix, Iy, It = get_gradients_optical_flow(img1, img2)
               
        iter_counter = 0

        while True:
            iter_counter += 1
            # Check if our iteration is on range.
            if iter_counter > its:
                break

            # Compute local averages of the flow vectors
            uAvg = cv2.filter2D(U, -1, kernel)
            vAvg = cv2.filter2D(V, -1, kernel)
            uNumer = np.multiply((np.multiply(Ix, uAvg) + np.multiply(Iy, vAvg) + It), Ix)
            uDenom = alpha ** 2 + np.power(Ix, 2) + np.power(Iy, 2)
            # U -> flow[:,:,0]
            U_prev = U
            U = uAvg - np.divide(uNumer, uDenom)

            vNumer = np.multiply((np.multiply(Ix, uAvg) + np.multiply(Iy, vAvg) + It), Iy)
            vDenom = alpha ** 2 + np.power(Ix, 2) + np.power(Iy, 2)
            # V -> flow[:,:,1]
            V = vAvg - np.divide(vNumer, vDenom)
            print("Iteration number: ", iter_counter)

            #If Check if our iteration is on range.
            if convergence:
                diff = np.linalg.norm(U - U_prev, 2)
                print("Iteration number: ", iter_counter, " / Error: ", diff, diff < delta)
                if diff < delta:
                    break

        flow[:, :, 0] = U
        flow[:, :, 1] = V
        return flow
    else:
        print("Error input data. Images must be in grayscale.")
        return

def remove_noise(flow_in, th = 0.1):
    dim = flow_in.shape
    flow_out = np.zeros_like(flow_in)
    #print(np.min(flow_in), np.max(flow_in))
    magnitude = np.linalg.norm(flow_in, axis=2)
    #plt.hist(magnitude)
    #print(np.min(magnitude), np.max(magnitude), np.abs(np.max(magnitude) - np.min(magnitude)))
    #min_prox = np.min(magnitude) + (th * (np.abs(np.max(magnitude) - np.min(magnitude))))
    #print(min_prox, np.min(magnitude), (th * (np.abs(np.max(magnitude) - np.min(magnitude)))))
    magnitude[magnitude < (th * (np.abs(np.max(magnitude) - np.min(magnitude))))] = 0

    for y in range(0, dim[0]):
        for x in range(0, dim[1]):
            if magnitude[y, x] != 0:
                flow_out[y, x, :] = flow_in[y, x, :]

    #print(np.min(flow_out), np.max(flow_out), np.abs(np.max(flow_out) - np.min(flow_out)))
    return flow_out


def plot_quiver(ax, flow, spacing=5, normalize=False, **kwargs):
    dim = flow.shape
    x = np.arange(0, dim[1], spacing)
    y = np.arange(0, dim[0], spacing)
    flow = flow[np.ix_(y, x)]
    X, Y = np.meshgrid(x, y)
    n = -5

    #Normalize the arrows:
    if normalize:
        eps = 0
        #eps = np.finfo(np.float32).eps
        U = np.divide(flow[:, :, 0], np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2) + eps)
        V = np.divide(flow[:, :, 1], np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2) + eps)
        U = np.nan_to_num(U)
        V = np.nan_to_num(V)
    else:
        U = flow[:, :, 0]
        V = flow[:, :, 1]

    color_array = np.sqrt(((V - n) / 2) ** 2 + ((U - n) / 2) ** 2)
    kwargs = {**dict(angles="xy", scale_units="xy"), **kwargs}
    ax.quiver(X, Y, U, V, color_array, **kwargs)
    # angles="xy", scale_units="xy", angles="uv", units="dots", angles="uv", scale_units="xy", scale=0.5, alpha=1, color="#ff44ff", units="dots", angles="uv", scale_units="dots"
    #ax.grid()
    ax.set_ylim(sorted(ax.get_ylim(), reverse=True))
    ax.set_aspect("equal")
    #ax.patch.set_alpha(0.15)
    #ax.patch.set_facecolor('gray')
    return ax


def plot_rgb(flow, img):
    dim = img.shape
    mask = np.zeros((dim[0], dim[1], 3))
    mask = mask.astype(np.uint8)
    mask[:, :, 1] = 255

    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
    # Sets image hue according to the optical flow direction
    mask[:, :, 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow magnitude (normalized)
    mask[:, :, 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    return rgb

def draw_optical_flow(flow, img2, quiver_plot=True, scale=None, color_plot=True):
    fig2, ax2 = plt.subplots(figsize=(10, 6), nrows=1, ncols=1)
    transp = 1
    if quiver_plot:
        plot_quiver(ax2, flow, 5, normalize=False)
    if color_plot:
        transp = 0.3
        rgb1 = plot_rgb(flow, img2)
        plt.imshow(rgb1, interpolation='none')
    plt.imshow(img2, cmap='gray', interpolation='none', alpha=transp)
    plt.show()
