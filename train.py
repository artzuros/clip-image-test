from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('yolov8l.yaml')  # Load model

    results = model.train(data='D:/C_Drive/Pictures/IIITD Dataset/data/config.yaml', epochs = 20)  # Train model
    # model.export('trained_model')  # Save model