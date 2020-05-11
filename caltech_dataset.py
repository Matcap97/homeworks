from torchvision.datasets import VisionDataset
from PIL import Image
from sklearn.preprocessing import LabelEncoder

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)
        self.split = split+'.txt' 
        image=[]
        label=[]
        path='./Caltech101/'+self.split
        with open(path,'r') as f:
            for i in f:
                i=i.rstrip('\n')
                if i[:-15]!='BACKGROUND_Google' and i!='BACKGROUND_Google/tmp':
                    im,l=(pil_loader(self.root+i),i[:-15])
                    image.append(im)
                    label.append(l)
        self.image=image
        le=LabelEncoder()
        self.label=le.fit_transform(label)
        
    def __getitem__(self, index):
        label=self.label[index]
        image=self.image[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        length = len(self.label)
        return length

    
