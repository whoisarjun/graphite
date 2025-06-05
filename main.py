import image_processor as ip
from graphite import Graphite

model = Graphite(model_path='graphite.pt')
ip.process('', 'output.png')
print(model.predict('output.png'))
