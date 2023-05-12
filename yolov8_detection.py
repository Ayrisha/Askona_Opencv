from ultralytics import YOLO
import numpy


def yolov8_detect(image):
    """
    yolov8_detect находит предметы на фото

        Параметры:
            image: фото

        Возвращаемое значение:
            boxes: ограничивающие рамки найденных предметов
            class_ids: id найденных предметов
            segmentation_contours_idx: сегментация найденных предметов

    """
    height, width, channels = image.shape
    model = YOLO("yolov/yolov8x-seg.pt")

    results = model(image)
    result = results[0]
    segmentation_contours_idx = []
    for seg in result.masks.segments:
        seg[:, 0] *= width
        seg[:, 1] *= height
        segment = numpy.array(seg, dtype=numpy.int32)
        segmentation_contours_idx.append(segment)

    boxes = numpy.array(result.boxes.xyxy)
    class_ids = numpy.array(result.boxes.cls)

    return boxes, class_ids, segmentation_contours_idx
