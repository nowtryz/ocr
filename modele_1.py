#
# ATTENTION : NE PAS METTRE D'ACCENT, MEME DANS LES COMMENTAIRES
#
# import des bibliotheques
from skimage import io
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np

class image:
    # Initialisation d'une image composee d'un tableau 2D vide
    # (pixels) et de 2 dimensions (H = height et W = width) mises a 0
    def __init__(self):
        self.pixels = None
        self.H = 0
        self.W = 0

    # Remplissage du tableau pixels de l'image self avec un tableau 2D (tab_pixels)
    # et affectation des dimensions de l'image self avec les dimensions
    # du tableau 2D (tab_pixels)
    def set_pixels(self, tab_pixels):
        self.pixels = tab_pixels
        self.H,self.W = self.pixels.shape

    # Lecture d'un image a partir d'un fichier de nom "file_name"
    def load_image(self, file_name):
        self.pixels = io.imread(file_name)
        self.H,self.W = self.pixels.shape
        print("lecture image : " + file_name + " (" + str(self.H) + "x" + str(self.W) + ")")

    # Affichage a l'ecran d'une image
    def display(self, window_name):
        fig=plt.figure(window_name)
        if (not (self.pixels is None)):
            io.imshow(self.pixels)
            io.show()
        else:
            print("L'image est vide")


    #==============================================================================
    # Methode mettant en blanc tous les pixels de l'image self
    # dont l'intensite est superieure ou egale au seuil S
    #==============================================================================

    def modif_ima(self,S):

        # preparaton du resultat : creation d'une image vide
        im_modif = image()
        # affectation de l'image resultat par un tableau de 0, de meme taille
        # que le tableau de pixels de l'image self
        # les valeurs sont de type uint8 (8bits non signes)
        im_modif.set_pixels(np.zeros((self.H,self.W), dtype=np.uint8))

        # boucle imbriquees pour parcourir tous les pixels de l'image
        for l in range(self.H):
            for c in range(self.W):
                # modif des pixels d'intensite >= a S
                if self.pixels[l][c] >= S:
                    im_modif.pixels[l][c] = 255
                else:
                    im_modif.pixels[l][c] = self.pixels[l][c]
        return im_modif

    def binaris(self, s):
        """
        Transforme cette image en noir et blanc
        :param s: Quantité de blanc pour considérer un pixel comme blanc
        :return: L'image modifiée
        """
        pixels = np.where(self.pixels < s, 0, 255)
        im_modif = image()
        im_modif.set_pixels(np.array(pixels, dtype=np.uint8))
        return im_modif

    def localisation(self):
        """
        Transforme cette image pour ne garder que la partie avec des pixels noirs
        :return: l'image modifiée
        """
        Xs, Ys = np.nonzero(self.pixels == 0)
        pixels = self.pixels[min(Xs):max(Xs), min(Ys):max(Ys)]
        im_modif = image()
        im_modif.set_pixels(np.array(pixels, dtype=np.uint8))
        return im_modif

#==============================================================================
# # Programme principal
#==============================================================================

# Lecure d'une image depuis un fichier
ima_test = image()
ima_test.load_image('test10.jpg')

# Affichage de cette image
ima_test.display("image initiale")

# Creation d'une image modifiee
seuil = 145
ima_modif = ima_test.modif_ima(seuil)


# Affichage de l'image modifiee
ima_modif.display("image modifiee")

binari = ima_test.binaris(seuil)
binari.display("binaris")

localisation = ima_test.binaris(seuil).localisation()
localisation.display("localisation")

print("fin du programme")
