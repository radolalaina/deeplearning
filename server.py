import tornado.ioloop
import tornado.web
import model
import uuid

import image_loader

__UPLOADS__ = "uploads"

m = model.DeepMatch()

class TestForm(tornado.web.RequestHandler):
    def get(self):
        self.write("<html><head></head><body><form  enctype='multipart/form-data' action='/predict' method='post'><input name='file' type='file' /><input type='submit' /></form></body></html>")

class DeepMatchHandler(tornado.web.RequestHandler):
    def post(self):
        fileinfo = self.request.files['file'][0]
        image = image_loader.from_string(fileinfo['body'])
        match = m.predict(image.reshape(1, 3, 128, 128))[0]
        self.finish(str(match))


def make_app():
    return tornado.web.Application([
        (r"/", TestForm),
        (r"/predict", DeepMatchHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
