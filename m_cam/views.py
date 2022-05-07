from django.shortcuts import render
from django.http import StreamingHttpResponse
from m_cam.camera import VideoCamera, IPWebCam

# Create your views here.
def home(request):
    #to render string as http response to the request
    #return HttpResponse("<h1>Hello World</h1>")
    #
    return render(request, 'home.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n\r\n')
    
def video_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()),
					content_type='multipart/x-mixed-replace; boundary=frame')


def webcam_feed(request):
	return StreamingHttpResponse(gen(IPWebCam()),
					content_type='multipart/x-mixed-replace; boundary=frame')
