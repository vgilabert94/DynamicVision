import cv2
import os
import numpy as np


def plot_stacked_imgs(img1, img2):

    result = np.concatenate((img1, img2), axis=1)
    result = resize_img(result, 2)
    cv2.imshow("Stacked images", result)
    cv2.waitKey(0)

def resize_img(img, scale_percent=0.5):
    '''
    resize image using scale_percent.
    '''
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def hsv_filter(img):
    lower1 = np.array([0, 80, 80])
    upper1 = np.array([5, 255, 255])
    lower2 = np.array([170, 80, 80])
    upper2 = np.array([180, 255, 255])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_mask = cv2.inRange(hsv, lower1, upper1)
    upper_mask = cv2.inRange(hsv, lower2, upper2)
    mask = lower_mask + upper_mask
    _, mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def draw_best_particle(img_in, particles, M, weight):

    if np.count_nonzero(weight) != 0:
        M_half = int(M / 2)
        best_particle = particles[np.argmax(weight)]
        drawed = cv2.rectangle(img_in, (best_particle[0]-M_half, best_particle[1]-M_half),
                               (best_particle[0]+M_half, best_particle[1]+M_half),
                               (0, 0, 255), 3)
        return drawed
    else:
        drawed = img_in
        return drawed


def draw_particles_img_debug(img_in, particles, M, weight=None):
    # np.count_nonzero(weight) == 0 -> Igual. Todos los valores son 0.
    # np.count_nonzero(weight) != 0 -> Diferente. Es decir tengo valores de peso validos.
    M_half = int(M / 2)
    drawed = img_in

    for i in range(0, len(particles)):
        x = particles[i][0]
        y = particles[i][1]
        drawed = cv2.rectangle(drawed, (x-M_half, y-M_half), (x+M_half, y+M_half), (255, 255, 0), 1)

    # Si tenemos pesos, pintamos de rojo la particula con mayor peso.
    if weight is not None:
        if np.count_nonzero(weight) != 0:
            idx = np.argmax(weight)
            max_w = particles[idx]
            drawed = cv2.rectangle(drawed, (max_w[0] - M_half, max_w[1] - M_half),
                                            (max_w[0] + M_half, max_w[1] + M_half), (0, 0, 255), 2)
    return drawed


def particles_init(img, N, M):
    dim = img.shape
    M_half = int(M / 2)
    new_particles = []
    vx = vy = 0
    
    for i in range(0, N):
        y = np.random.randint(M_half, dim[0] - M_half)
        x = np.random.randint(M_half, dim[1] - M_half)
        new_particles.append([x, y, vx, vy])
    return np.array(new_particles)


def particles_weight(img, particles, M):
    weights = []
    M_half = int(M / 2)

    for x, y, _, _ in particles:
        if np.sum(img) != 0:
            y1 = y - M_half
            y2 = y + M_half
            x1 = x - M_half
            x2 = x + M_half
            weights.append(np.sum(img[y1:y2, x1:x2]) / (4*M_half*M_half))
        else:
            weights.append(0)

    if np.sum(weights) != 0:
        # normalizar los pesos.
        weights = weights / np.sum(weights)

    # Obtener pesos acumulados.
    weights_cum = np.cumsum(weights)

    return np.nan_to_num(weights), np.nan_to_num(weights_cum)

#def estimation_stage(particles, weights):
#
#    best_particle = particles[np.argmax(weights)]
#    w_particles = particles[weights > 0.01]
#    return best_particle, w_particles


def selection_stage(particles, weights_cum):
    new_particles = []

    for _ in range(len(particles)):
        rdm = np.random.uniform()
        idx = np.argmin(rdm > weights_cum)
        new_particles.append(particles[idx])

    return np.array(new_particles)


def diffusion(img, particles, w, M, std=10):
    new_particles = []
    # Elitismo. Nos quedamos con la mejor particula antes de aplicar la difusion sobre el resto.
    new_particles.append(particles[np.argmax(w)])

    for i in range(len(particles)-1):
        x, y, vx, vy = particles[i]
        x_offset = int(np.random.normal(0, std))
        y_offset = int(np.random.normal(0, std))
        new_particles.append([x+x_offset, y+y_offset, vx, vy])

    # Funcion para evitar las que las partículas se salgan fuera de la imagen despues de la difusión.
    new_particles = particles_checkPosition(img, new_particles, M)
    return np.array(new_particles)


def prediction(img, particles, M, std=5):
    new_particles = []

    for i in range(len(particles)):
        x, y, vx, vy = particles[i]
        vx_offset = int(np.random.normal(0, std))
        vy_offset = int(np.random.normal(0, std))
        # Estimacion de movimiento.
        # vx(t+1) = vx(t) + N(0,std) // vy(t+1) = vy(t) + N(0,std)
        vx_new = vx + vx_offset
        vy_new = vy + vy_offset
        # Actualizamos la posicion de la particula con el movimiento anterior.
        # x(t+1) = x(t) + vx(t+1) // y(t+1) = y(t) + vy(t+1)
        x_new = x + vx_new
        y_new = y + vy_new
        new_particles.append([x_new, y_new, vx_new, vy_new])

    # Funcion para evitar las que las partículas se salgan fuera de la imagen despues de la prediccion.
    new_particles = particles_checkPosition(img, new_particles, M)
    return np.array(new_particles)

def particles_checkPosition(img, particles, M):
    dim = img.shape
    M_half = int(M / 2)
    new_particles = []

    for x, y, vx, vy in particles:
        x_new = x
        y_new = y
        x1 = x - M_half
        y1 = y - M_half
        x2 = x + M_half
        y2 = y + M_half
        if x1 < 0: x_new = abs(x) + M_half + 1
        if y1 < 0: y_new = abs(y) + M_half + 1
        if x2 > dim[1]: x_new = dim[1] - M_half - 1
        if y2 > dim[0]: y_new = dim[0] - M_half - 1
        #print([x, y], [x1, y1, x2, y2], [x_new, y_new])
        new_particles.append([x_new, y_new, vx, vy])

    return new_particles


def main():

    path = "video/"
    N = 100     # Numero de particulas
    M = 40      # Tamaño de la particula
    instante_inicial = True
    particles = np.zeros((N, 4))
    frame = 0

    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                frame += 1
                img_path = os.path.join(path, file)
                img = cv2.imread(img_path)
                mask = hsv_filter(img)
                #plot_stacked_imgs(img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

                if instante_inicial:
                    particles = particles_init(img, N, M)
                    w, w_cum = particles_weight(mask/255., particles, M)
                    draw_init = draw_particles_img_debug(img.copy(), particles, M, weight=w)
                    if np.count_nonzero(w) == 0:
                        instante_inicial = True
                        draw = img
                    else:
                        draw = draw_best_particle(img.copy(), particles, M, w)
                        particles = selection_stage(particles, w_cum)
                        instante_inicial = False

                else:
                    w, w_cum = particles_weight(mask/255., particles, M)

                    if np.count_nonzero(w) == 0:
                        instante_inicial = True
                        draw = img

                    else:
                        draw = draw_best_particle(img.copy(), particles, M, w)
                        particles = selection_stage(particles, w_cum)
                        particles = diffusion(img, particles, w, M, 20)
                        particles = prediction(img, particles, M, 15)

                text = "Frame: " + str(frame)
                cv2.putText(img=draw, text=text, org=(20, 20), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5,
                            color=(0, 255, 0), thickness=1)
                cv2.imshow("Particles FINAL", draw)
                #cv2.imshow("Particles INIT", draw_init)
                cv2.waitKey(0)

    else:
        print("Error con la ruta.")


if __name__ == '__main__':
    main()


# DEBUG
#draw_init = draw_particles_img_debug(img.copy(), particles, M)
#draw_selection = draw_particles_img_debug(img.copy(), particles, M)
#draw_diffusion = draw_particles_img_debug(img.copy(), particles, M)
# draw_prediction = draw_particles_img_debug(img.copy(), particles, M)
# cv2.imshow("Particles SELECTION", draw_selection)
# cv2.imshow("Particles DIFFUSION", draw_diffusion)
# cv2.imshow("Particles PREDICTION", draw_prediction)
# cv2.waitKey(100)
