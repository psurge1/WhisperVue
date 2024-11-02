from flask import Flask, render_template, Response
import cv2

class VideoCamera(object):
    def __init__(self):
      self.video = cv2.VideoCapture(0)
    def __del__(self):
      self.video.release()
    def get_frame(self):
      ret, frame = self.video.read()
      ret, jpeg = cv2.imencode('.jpg', frame)
      return jpeg.tobytes()

app = Flask(__name__)
# @app.route('/')
# def index():
#     return render_template('cam.js')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)