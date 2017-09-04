from pics import *
from neural import *
from utils import *

path = "./weightmatrix-tests/testimage.jpg"

# pic = bipolize_image(path, 150)
pic = binarize_image(path, 150)
ShowBinayImage(pic)

flatpat = pic.flatten()
connectionpattern = np.outer(flatpat,flatpat)
np.fill_diagonal(connectionpattern,0)

ShowBinayImage(connectionpattern)