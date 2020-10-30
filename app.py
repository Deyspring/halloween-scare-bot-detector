import time
import edgeiq
import post
"""
Use object detection to detect objects in the frame in realtime. The
types of objects detected can be changed by selecting different models.

To change the computer vision model, follow this guide:
https://alwaysai.co/docs/application_development/changing_the_model.html

To change the engine and accelerator, follow this guide:
https://alwaysai.co/docs/application_development/changing_the_engine_and_accelerator.html

To install app dependencies in the runtime container, list them in the
requirements.txt file.
"""

SERVER_URL='http://0.0.0.0:5000/aai_track'
ENABLE_STREAMER = False
ENABLE_SEND = True

def is_accelerator_available():
    """Detect if an Intel Neural Compute Stick accelerator is attached"""
    if edgeiq.find_ncs2():
        return True
    return False

def engine():
    """Switch Engine modes if an Intel accelerator is available"""
    if is_accelerator_available() == True:
        return edgeiq.Engine.DNN_OPENVINO
    return edgeiq.Engine.DNN


def main():
    obj_detect = edgeiq.ObjectDetection("alwaysai/mobilenet_ssd")
    obj_detect.load(engine())
    # obj_detect.load(engine=edgeiq.Engine.DNN)

    print("Loaded model:\n{}\n".format(obj_detect.model_id))
    print("Engine: {}".format(obj_detect.engine))
    print("Accelerator: {}\n".format(obj_detect.accelerator))
    print("Labels:\n{}\n".format(obj_detect.labels))

    fps = edgeiq.FPS()

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()
            width = 0
            height = 0

            # loop detection
            while True:
                frame = video_stream.read()
                if width == 0:
                    height, width, _ = frame.shape
                results = obj_detect.detect_objects(frame, confidence_level=.5)
                frame = edgeiq.markup_image(
                        frame, results.predictions, colors=obj_detect.colors)

                # Generate text to display on streamer
                text = ["Model: {}".format(obj_detect.model_id)]
                text.append(
                        "Inference time: {:1.3f} s".format(results.duration))
                text.append("Objects:")

                predictions = edgeiq.filter_predictions_by_label(
                                    results.predictions, ["person"])

                largest_area = 0
                larget_prediction = None
                for prediction in predictions:
                    if prediction.label == "person":
                        if prediction.box.area > largest_area:
                            larget_prediction = prediction
                        text.append("{}: {:2.2f}%: center:{} area:{}".format(
                            prediction.label, prediction.confidence * 100, prediction.box.center, prediction.box.area))
                
                text.append("Frame width:{} height:{}".format(width, height))

                # Send data to server
                if ENABLE_SEND:
                    payload = {"X": larget_prediction.box.center[0], 
                                "Y":larget_prediction.box.center[1], 
                                "W": width,
                                "H": height}
                    post.data(SERVER_URL,payload)

                if ENABLE_STREAMER:
                    streamer.send_data(frame, text)
                    fps.update()
                    if streamer.check_exit():
                        break

    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))
        print("Program Ending")


if __name__ == "__main__":
    main()
