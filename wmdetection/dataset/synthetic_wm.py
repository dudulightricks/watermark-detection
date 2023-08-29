import os
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import cv2
import string 
import random

CV2_FONTS = [
    #cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_ITALIC,
    cv2.QT_FONT_BLACK,
    cv2.QT_FONT_NORMAL
]

# рандомный float между x и y
def random_float(x, y):
    return random.random()*(y-x)+x

# вычисляет размер текста в пикселях для cv2.putText
def get_text_size(text, font, font_scale, thickness):
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    return w, h+baseline

# вычисляет какой нужен font_scale для определенного размера текста (по высоте)
def get_font_scale(needed_height, text, font, thickness):
    w, h = get_text_size(text, font, 1, thickness)
    return needed_height/h

# добавляет текст на изображение
def place_text(image, text, color=(255,255,255), alpha=1, position=(0, 0), angle=0,
               font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.0, thickness=3):
    image = np.array(image)
    overlay = np.zeros_like(image)
    output = image.copy()
    
    cv2.putText(overlay, text, position, font, font_scale, color, thickness)
    
    if angle != 0:
        text_w, text_h = get_text_size(text, font, font_scale, thickness)
        rotate_M = cv2.getRotationMatrix2D((position[0]+text_w//2, position[1]-text_h//2), angle, 1)
        overlay = cv2.warpAffine(overlay, rotate_M, (overlay.shape[1], overlay.shape[0]))


    arr_gray = np.average(overlay, weights=[0.2989, 0.5870, 0.1140], axis=2)
    y_indices, x_indices = np.where(arr_gray > 0)
    center_y = np.average(y_indices, weights=arr_gray[arr_gray > 0]) / arr_gray.shape[0]
    center_x = np.average(x_indices, weights=arr_gray[arr_gray > 0]) / arr_gray.shape[1]
    width = (np.max(x_indices) - np.min(x_indices)) / arr_gray.shape[1]
    height = (np.max(y_indices) - np.min(y_indices)) / arr_gray.shape[0]
    center_data = [(center_x, center_y, width, height)]

    overlay[overlay==0] = image[overlay==0]
    cv2.addWeighted(overlay, alpha, output, 1-alpha, 0, output)
    
    return Image.fromarray(output), center_data

def get_random_font_params(text, text_height, fonts, font_thickness_range):
    font = random.choice(fonts)
    font_thickness_range_scaled = [int(font_thickness_range[0]*(text_height/35)),
                                   int(font_thickness_range[1]*(text_height/85))]
    try:
        font_thickness = min(random.randint(*font_thickness_range_scaled), 2)
    except ValueError:
        font_thickness = 2
    font_scale = get_font_scale(text_height, text, font, font_thickness)
    return font, font_scale, font_thickness

# устанавливает вотермарку в центре изображения с рандомными параметрами
def place_random_centered_watermark(
        pil_image, 
        text,
        center_point_range_shift=(-0.025, 0.025),
        random_angle=(0,0),
        text_height_in_percent_range=(0.15, 0.18),
        text_alpha_range=(0.23, 0.5),
        fonts=CV2_FONTS,
        font_thickness_range=(2, 7),
        colors=[(255,255,255)]
    ):
    w, h = pil_image.size
    
    position_shift_x = random_float(*center_point_range_shift)
    offset_x = int(w*position_shift_x)
    position_shift_y = random_float(*center_point_range_shift)
    offset_y = int(w*position_shift_y)
    
    text_height = int(h*random_float(*text_height_in_percent_range))
    
    font, font_scale, font_thickness = get_random_font_params(text, text_height, fonts, font_thickness_range)

    text_width, _ = get_text_size(text, font, font_scale, font_thickness)
    
    position_x = int((w/2)-text_width/2+offset_x)
    position_y = int((h/2)+text_height/2+offset_y)
    
    return place_text(
        pil_image, 
        text,
        color=random.choice(colors),
        alpha=random_float(*text_alpha_range),
        position=(position_x, position_y), 
        angle=random.randint(*random_angle),
        thickness=font_thickness,
        font=font, 
        font_scale=font_scale
    )


def place_random_centered_getty_watermark(
    pil_image,
    text,
    center_point_range_shift=(-0.025, 0.025),
    random_angle=(0, 0),
    text_alpha_range=(0.23, 0.5),
    watermark_bg_color_range=(0.3, 0.7),
    watermark_bg_alpha_range=(0.2, 0.3),
):
    w, h = pil_image.size
    color = random_float(*watermark_bg_color_range)
    background_alpha = random_float(*watermark_bg_alpha_range)

    position_shift_x = random_float(*center_point_range_shift)
    offset_x = int(w * position_shift_x)
    position_shift_y = random_float(*center_point_range_shift)
    offset_y = int(w * position_shift_y)

    watermark = create_getty_primary_text_watermark(
        500, 85, text, background_alpha, background_gray_level=color, inner_scale=random_float(0.5, 1)
    )
    num = ''.join(["{}".format(random.randint(0, 9)) for _ in range(0, 10)])
    secondary_watermark = create_getty_secondary_text_watermark(
        95, 18, num, background_alpha, background_gray_level=color, inner_scale=random_float(0.5, 1)
    )

    h_wm, w_wm, _ = watermark.shape
    wm_new_width = random.randint(100, w // 2)
    wm_new_height = int(h_wm * (wm_new_width / w_wm))
    watermark = cv2.resize(watermark, (wm_new_width, wm_new_height), interpolation=cv2.INTER_AREA)
    position_x = int((w / 2) - wm_new_width / 2 + offset_x)
    position_y = int((h / 2) + wm_new_height / 2 + offset_y)

    h_wm2, w_wm2, _ = secondary_watermark.shape
    wm2_new_width = random.randint(w // 12, w // 10)
    wm2_new_height = int(h_wm2 * (wm2_new_width / w_wm2))
    secondary_watermark = cv2.resize(secondary_watermark, (wm2_new_width, wm2_new_height),
                                     interpolation=cv2.INTER_AREA)
    position_x2 = random.randint(0, max(w - wm2_new_width, 10))
    position_y2 = random.randint(0, max(h - wm2_new_height, 10))

    return place_getty_watermark(
        pil_image,
        watermark,
        secondary_watermark,
        alpha=random_float(*text_alpha_range),
        position=(position_x, position_y),
        position2=(position_x2, position_y2),
        angle=random.randint(*random_angle),
    )


def place_random_getty_watermark(
        pil_image,
        text,
        random_angle=(0, 0),
        text_alpha_range=(0.18, 0.4),
        watermark_bg_color_range=(0.3, 0.7),
        watermark_bg_alpha_range=(0.2, 0.3),
):
    w, h = pil_image.size
    color = random_float(*watermark_bg_color_range)
    background_alpha = random_float(*watermark_bg_alpha_range)
    watermark = create_getty_primary_text_watermark(
        500, 85, text, background_alpha, background_gray_level=color, inner_scale=random_float(0.5, 1)
    )
    num = ''.join(["{}".format(random.randint(0, 9)) for _ in range(0, 10)])
    secondary_watermark = create_getty_secondary_text_watermark(
        95, 18, num, background_alpha, background_gray_level=color, inner_scale=random_float(0.5, 1)
    )

    h_wm, w_wm, _ = watermark.shape
    wm_new_width = random.randint(3 * w // 8, 5 * w // 8)
    wm_new_height = int(h_wm * (wm_new_width / w_wm))
    watermark = cv2.resize(watermark, (wm_new_width, wm_new_height), interpolation=cv2.INTER_AREA)
    position_x = random.randint(0, max(w - wm_new_width, 10))
    position_y = random.randint(0, max(h - wm_new_height, 10))

    h_wm2, w_wm2, _ = secondary_watermark.shape
    wm2_new_width = random.randint(w // 12, w // 10)
    wm2_new_height = int(h_wm2 * (wm2_new_width / w_wm2))
    secondary_watermark = cv2.resize(secondary_watermark, (wm2_new_width, wm2_new_height),
                                     interpolation=cv2.INTER_AREA)
    position_x2 = random.randint(0, max(w - wm2_new_width, 10))
    position_y2 = random.randint(0, max(h - wm2_new_height, 10))

    return place_getty_watermark(
        pil_image,
        watermark,
        secondary_watermark,
        alpha=random_float(*text_alpha_range),
        position=(position_x, position_y),
        position2=(position_x2, position_y2),
        angle=random.randint(*random_angle),
    )

def place_random_centered_shutterstock_watermark(
        pil_image,
        center_point_range_shift=(-0.025, 0.025),
        random_angle=(0, 0),
        text_alpha_range=(0.23, 0.5),
):
    w, h = pil_image.size

    position_shift_x = random_float(*center_point_range_shift)
    offset_x = int(w * position_shift_x)
    position_shift_y = random_float(*center_point_range_shift)
    offset_y = int(w * position_shift_y)

    watermark = Image.open("wmdetection/dataset/shutterstock-logo.png")
    watermark = np.array(watermark).astype(np.float32)
    watermark = np.clip(watermark * random_float(0.8, 2.0), 0, 255)
    watermark = np.stack([watermark] * 3, axis=-1)

    h_wm, w_wm, _ = watermark.shape
    wm_new_width = random.randint(w // 3, 2 * w // 3)
    wm_new_height = int(h_wm * (wm_new_width / w_wm))
    watermark = cv2.resize(watermark, (wm_new_width, wm_new_height), interpolation=cv2.INTER_AREA)
    position_x = int((w / 2) - wm_new_width / 2 + offset_x)
    position_y = int((h / 2) + wm_new_height / 2 + offset_y)

    return place_shutterstock_watermark(
        pil_image,
        watermark,
        alpha=random_float(*text_alpha_range),
        position=(position_x, position_y),
        angle=random.randint(*random_angle),
    )

def place_random_watermark(
        pil_image, 
        text,
        random_angle=(0,0),
        text_height_in_percent_range=(0.10, 0.18),
        text_alpha_range=(0.18, 0.4),
        fonts=CV2_FONTS,
        font_thickness_range=(2, 6),
        colors=[(255,255,255)]
    ):
    w, h = pil_image.size
    
    text_height = int(h*random_float(*text_height_in_percent_range))
    
    font, font_scale, font_thickness = get_random_font_params(text, text_height, fonts, font_thickness_range)

    text_width, _ = get_text_size(text, font, font_scale, font_thickness)
    
    position_x = random.randint(0, max(w-text_width, 10))
    position_y = random.randint(text_height, h)
    
    return place_text(
            pil_image, 
            text,
            color=random.choice(colors),
            alpha=random_float(*text_alpha_range),
            position=(position_x, position_y), 
            angle=random.randint(*random_angle),
            thickness=font_thickness,
            font=font, 
            font_scale=font_scale
        )

def center_crop(image, w, h):
    center = image.shape
    x = center[1]/2 - w/2
    y = center[0]/2 - h/2
    return image[int(y):int(y+h), int(x):int(x+w)]

# добавляет текст в шахматном порядке на изображение
def place_text_checkerboard(image, text, color=(255,255,255), alpha=1, step_x=0.1, step_y=0.1, angle=0,
                            font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.0, thickness=3):
    image_size = image.size
    
    image = np.array(image.convert('RGB'))
    if angle != 0:
        border_scale = 0.4
        overlay_size = [int(i*(1+border_scale)) for i in list(image_size)]
    else:
        overlay_size = image_size
        
    w, h = overlay_size
    overlay = np.zeros((overlay_size[1], overlay_size[0], 3)) # change dimensions
    output = image.copy()
    
    text_w, text_h = get_text_size(text, font, font_scale, thickness)

    if angle != 0:
        rotate_M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)

    c = 0
    center_data = []
    for rel_pos_x in np.arange(0, 1, step_x):
        c += 1
        for rel_pos_y in np.arange(text_h/h+(c%2)*step_y/2, 1, step_y):
            position = (int(w*rel_pos_x), int(h*rel_pos_y))
            temp_overlay = np.zeros((overlay_size[1], overlay_size[0], 3))
            cv2.putText(temp_overlay, text, position, font, font_scale, color, thickness)
            if angle != 0:
                temp_overlay = cv2.warpAffine(temp_overlay, rotate_M, (temp_overlay.shape[1], temp_overlay.shape[0]))
            temp_overlay = center_crop(temp_overlay, image_size[0], image_size[1])
            arr_gray = np.average(temp_overlay, weights=[0.2989, 0.5870, 0.1140], axis=2)
            y_indices, x_indices = np.where(arr_gray > 0)
            if np.sum(y_indices) > 0 and np.sum(x_indices) > 0:
                center_y = np.average(y_indices, weights=arr_gray[arr_gray > 0]) / arr_gray.shape[0]
                center_x = np.average(x_indices, weights=arr_gray[arr_gray > 0]) / arr_gray.shape[1]
                width = (np.max(x_indices) - np.min(x_indices)) / arr_gray.shape[1]
                height = (np.max(y_indices) - np.min(y_indices)) / arr_gray.shape[0]
                center_data.append((center_x, center_y, width, height))

            cv2.putText(overlay, text, position, font, font_scale, color, thickness)

    if angle != 0:
        overlay = cv2.warpAffine(overlay, rotate_M, (overlay.shape[1], overlay.shape[0]))

    
    overlay = center_crop(overlay, image_size[0], image_size[1])
    overlay[overlay==0] = image[overlay==0]
    overlay = overlay.astype(np.uint8)
    cv2.addWeighted(overlay, alpha, output, 1-alpha, 0, output)

    return Image.fromarray(output), center_data


def place_random_diagonal_watermark(
        pil_image, 
        text,
        random_step_x=(0.25, 0.4),
        random_step_y=(0.25, 0.4),
        random_angle=(-60,60),
        text_height_in_percent_range=(0.10, 0.18),
        text_alpha_range=(0.18, 0.4),
        fonts=CV2_FONTS,
        font_thickness_range=(2, 6),
        colors=[(255,255,255)]
    ):
    w, h = pil_image.size
    
    text_height = int(h*random_float(*text_height_in_percent_range))
    
    font, font_scale, font_thickness = get_random_font_params(text, text_height, fonts, font_thickness_range)

    text_width, _ = get_text_size(text, font, font_scale, font_thickness)
    
    return place_text_checkerboard(
            pil_image, 
            text,
            color=random.choice(colors),
            alpha=random_float(*text_alpha_range),
            step_x=random_float(*random_step_x),
            step_y=random_float(*random_step_y),
            angle=random.randint(*random_angle),
            thickness=font_thickness,
            font=font, 
            font_scale=font_scale
        )

def create_getty_primary_text_watermark(
        width, height, text, background_alpha, background_gray_level=0, inner_scale=1.0,
        watermark_path="wmdetection/dataset/getty-images-1-logo.png"
):
    width = int(inner_scale * width)
    height = int(inner_scale * height)
    bg_color = 255.0 * background_gray_level
    background_color = (bg_color, bg_color, bg_color, 255 * background_alpha)
    canvas = np.full(shape=(height, width, 4), fill_value=background_color).astype(np.float32)
    watermark = Image.open(watermark_path)
    watermark = np.array(watermark).astype(np.float32)
    mask = watermark[:, :, 0] > 0
    watermark[mask] = 255
    watermark[~mask] = 0

    original_watermark_h, original_watermark_w, _ = watermark.shape
    target_watermark_height = int(0.6 * height)
    target_watermark_width = int((target_watermark_height / original_watermark_h) * original_watermark_w)
    watermark = cv2.resize(np.array(watermark), (target_watermark_width, target_watermark_height), interpolation=cv2.INTER_AREA)
    watermark_top, watermark_left = 0, int(5 * inner_scale)
    watermark_bottom = watermark_top + target_watermark_height
    watermark_right = watermark_left + target_watermark_width
    canvas[watermark_top: watermark_bottom, watermark_left: watermark_right] += watermark

    text_height_in_percent_range = (0.15, 0.23)
    text_height = int(height * random_float(*text_height_in_percent_range))
    fonts = CV2_FONTS
    font_thickness_range = (2, 6)

    color = (255.0, 255.0, 255.0, 255.0)
    font, font_scale, font_thickness = get_random_font_params(text, text_height, fonts, font_thickness_range)
    cv2.putText(canvas, text, (watermark_left + int(7 * inner_scale), watermark_bottom + int(10 * inner_scale)),
                font, font_scale, color, font_thickness)

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    return canvas

def create_getty_secondary_text_watermark(width, height, text, background_alpha, background_gray_level=0, inner_scale=1.0):
    width = int(inner_scale * width)
    height = int(inner_scale * height)
    bg_color = 255.0 * background_gray_level
    background_color = (bg_color, bg_color, bg_color, 255 * background_alpha)
    canvas = np.full(shape=(height, width, 4), fill_value=background_color).astype(np.float32)

    text_height_in_percent_range = (0.6, 0.7)
    text_height = int(height * random_float(*text_height_in_percent_range))
    fonts = CV2_FONTS
    font_thickness_range = (1, 2)

    color = (255.0, 255.0, 255.0, 255.0)
    font, font_scale, font_thickness = get_random_font_params(text, text_height, fonts, font_thickness_range)
    text_width, text_height = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    cv2.putText(canvas, text, (width - text_width - int(5 * inner_scale), (height + text_height) // 2), font, font_scale, color)

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    return canvas

def place_shutterstock_watermark(image, watermark, alpha=1, position=(0, 0), angle=0):
    image = np.array(image).astype(np.float32)
    h_wm, w_wm, c_wm = watermark.shape

    overlay = np.zeros_like(image).astype(np.float32)
    output = image.copy().astype(np.float32)
    overlay[position[1]:position[1] + h_wm, position[0]:position[0] + w_wm] = watermark

    if angle != 0:
        rotate_M = cv2.getRotationMatrix2D((position[0] + w_wm // 2, position[1] + h_wm // 2), angle, 1)
        overlay = cv2.warpAffine(overlay, rotate_M, (overlay.shape[1], overlay.shape[0]))

    center_data = get_center_data_for_overlay(overlay, weights=[0.2989, 0.5870, 0.1140])
    overlay[overlay == 0] = image[overlay == 0]
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)), [center_data]

def place_getty_watermark(image, watermark, secondary_watermark, alpha=1, position=(0, 0), position2=(200, 200), angle=0):
    image = np.array(image).astype(np.float32)
    h_wm, w_wm, c_wm = watermark.shape
    h_wm2, w_wm2, c_wm2 = secondary_watermark.shape

    overlay = np.zeros_like(image).astype(np.float32)
    if overlay.shape[2] == 3:
        alpha_channel = np.ones(overlay.shape[:2], dtype=overlay.dtype) * 255
        overlay = np.dstack((overlay, alpha_channel))
    overlay2 = np.zeros_like(overlay).astype(np.float32)

    output = image.copy().astype(np.float32)
    overlay[position[1]:position[1] + h_wm, position[0]:position[0] + w_wm] = watermark
    overlay2[position2[1]:position2[1] + h_wm2, position2[0]:position2[0] + w_wm2] = secondary_watermark

    if angle != 0:
        rotate_M = cv2.getRotationMatrix2D((position[0] + w_wm // 2, position[1] + h_wm // 2), angle, 1)
        overlay = cv2.warpAffine(overlay, rotate_M, (overlay.shape[1], overlay.shape[0]))
        rotate_M2 = cv2.getRotationMatrix2D((position2[0] + w_wm2 // 2, position2[1] + h_wm2 // 2), angle, 1)
        overlay2 = cv2.warpAffine(overlay2, rotate_M2, (overlay2.shape[1], overlay2.shape[0]))

    center_data1 = get_center_data_for_overlay(overlay)
    center_data2 = get_center_data_for_overlay(overlay2)

    final_overlay = overlay + overlay2

    # If the overlay has an alpha channel, apply it to the RGB channels
    if final_overlay.shape[2] == 4 and image.shape[2] == 3:
        alpha_channel_overlay = final_overlay[:, :, 3] / 255.0
        final_overlay[:, :, :3] = final_overlay[:, :, :3] * alpha_channel_overlay[:, :, np.newaxis]
        final_overlay = final_overlay[:, :, :3]

    final_overlay[final_overlay == 0] = image[final_overlay == 0]
    cv2.addWeighted(final_overlay, alpha, output, 1 - alpha, 0, output)

    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)), [center_data1, center_data2]


def get_center_data_for_overlay(overlay, weights=[0.2989, 0.5870, 0.1140, 0.0]):
    arr_gray = np.average(overlay, weights=weights, axis=2)
    y_indices, x_indices = np.where(arr_gray > 0)
    center_y = np.average(y_indices, weights=arr_gray[arr_gray > 0]) / arr_gray.shape[0]
    center_x = np.average(x_indices, weights=arr_gray[arr_gray > 0]) / arr_gray.shape[1]
    width = (np.max(x_indices) - np.min(x_indices)) / arr_gray.shape[1]
    height = (np.max(y_indices) - np.min(y_indices)) / arr_gray.shape[0]
    return (center_x, center_y, width, height)


# img = create_getty_primary_text_watermark(500, 85, "some photographer", 0.2, watermark_path="getty-images-1-logo.png")
# img = Image.fromarray(img)
# img.save("/opt/watermark-detection/dataset/xx/fdfdfg.png")
#
# img2 = create_getty_secondary_text_watermark(95, 18, "1234567890", 0.3, background_gray_level=0.6)
# img2 = Image.fromarray(img2)
# img2.save("/opt/watermark-detection/dataset/xx/fdfdfg2.png")
