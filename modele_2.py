#
# ATTENTION : NE PAS METTRE D'ACCENT, MEME DANS LES COMMENTAIRES
#
# import des bibliotheques
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import sys


class image:
    # Initialisation d'une image composee d'un tableau 2D vide
    # (pixels) et de 2 dimensions (H = height et W = width) mises a 0
    def __init__(self, pixels=None):
        self.pixels = pixels
        if pixels is None:
            self.H = 0
            self.W = 0
        else:
            self.H, self.W = pixels.shape

    # Remplissage du tableau pixels de l'image self avec un tableau 2D (tab_pixels)
    # et affectation des dimensions de l'image self avec les dimensions
    # du tableau 2D (tab_pixels)
    def set_pixels(self, tab_pixels):
        self.pixels = tab_pixels
        self.H, self.W = self.pixels.shape

    # Lecture d'un image a partir d'un fichier de nom "file_name"
    def load_image(self, file_name, ecrire_nom=True):
        if ecrire_nom:
            print("lecture image : " + file_name, end=" ")
        self.pixels = io.imread(file_name)
        self.H, self.W = self.pixels.shape
        if ecrire_nom:
            print("(" + str(self.H) + "x" + str(self.W) + ")")

    # Affichage a l'ecran d'une image
    def display(self, window_name):
        fig = plt.figure(window_name)
        if (not (self.pixels is None)):
            io.imshow(self.pixels)
            io.show()
        else:
            print("L'image est vide")

    # ==============================================================================
    # Methode de binarisation
    # 2 parametres :
    #   self : l'image a binariser
    #   S : le seuil de binarisation
    #   on retourne une nouvelle image binarisee
    # ==============================================================================

    def binaris(self, s):
        """
        Transforme cette image en noir et blanc
        :param s: Seuil de blanc pour considerer un pixel comme blanc
        :return: L'image modifiee
        """
        pixels = np.where(self.pixels < s, 0, 255)
        return image(np.uint8(pixels))

    def binaris2(self, s):
        # preparaton du resultat : creation d'une image vide
        im_modif = image()
        # affectation de l'image resultat par un tableau de 0, de meme taille
        # que le tableau de pixels de l'image self
        # les valeurs sont de type uint8 (8bits non signes)
        im_modif.set_pixels(np.zeros((self.H, self.W), dtype=np.uint8))

        # boucle imbriquees pour parcourir tous les pixels de l'image
        for l in range(self.H):
            for c in range(self.W):
                # modif des pixels d'intensite >= a s en blanc
                if self.pixels[l][c] >= s:
                    im_modif.pixels[l][c] = 255
                # modif des pixels d'intensite < a s en noir
                else:
                    im_modif.pixels[l][c] = 0
        return im_modif

    # ==============================================================================
    # Dans une image binaire contenant une forme noire sur un fond blanc
    # la methode 'localisation' permet de limiter l'image au rectangle englobant
    # la forme noire
    # 1 parametre :
    #   self : l'image binaire que l'on veut recadrer
    #   on retourne une nouvelle image recadree
    # ==============================================================================

    def localisation(self):
        """
        Transforme cette image pour ne garder que la partie avec des pixels noirs
        :return: l'image modifiee
        """
        Xs, Ys = np.nonzero(self.pixels == 0)
        pixels = self.pixels[min(Xs):max(Xs) + 1, min(Ys):max(Ys) + 1]
        return image(pixels.astype(np.uint8))

    def localisation2(self):
        Xmin, Ymin = 0,0
        Xmax, Ymax = self.W, self.H

        project_vert = [sum(self.pixels[y]) for y in range(self.H)]
        for y in project_vert:
            if y != 0:
                Ymin = y
                break

        project_vert.reverse()
        for y in project_vert:
            if y != 0:
                Ymax = y
                break

        transposee = np.transpose(self.pixels)
        project_horiz = [sum(transposee[x]) for x in range(self.W)]
        for x in project_horiz:
            if x != 0:
                Xmin = x
                break

        project_horiz.reverse()
        for x in project_horiz:
            if x != 0:
                Xmax = x
                break

        pixels = self.pixels[Xmin:Xmax + 1, Ymin:Ymax + 1]
        return image(pixels.astype(np.uint8))


    # ==============================================================================
    # Methode de redimensionnement d'image
    # ==============================================================================

    def resize_im(self, new_H, new_W):
        """
        Redimensionne cette image aux dimension fournies
        :param new_H: nouvelle longueur
        :param new_W: nouvelle largeur
        :return: image modifie
        """
        pixels = resize(self.pixels, (new_H, new_W), 0) * 255
        return image(pixels.astype(np.uint8))

    # ==============================================================================
    # Methode de mesure de similitude entre l'image self et un modele im
    # ==============================================================================

    def simil_im(self, other):
        # si les tailles des images sont differentes alors les images sont differentes donc la similitude est nulle
        if self.pixels.shape != other.pixels.shape:
            return 0

        unique, counts = np.unique(self.pixels - other.pixels, return_counts=True)
        # on utilise la methode unique qui renvoie une liste avec les valeurs uniques et une autres tableau avec les
        # nombres de fois que les valeurs apparait
        values = dict(zip(unique, counts))
        # on met en dictionnaire ces deux listes ainsi on pourra recuperer le nombre de fois que la valeur apparait quand
        # on appelle cette valeur

        if 0 in values:
            return values[0] / (self.H * self.W) # on retourne
        return 0

    # ==============================================================================
    # Methode de localisation de differents caractères sur une ligne
    # ==============================================================================

    def localisation_ligne(self):
        transposee = np.transpose(255 - self.pixels)
        project_horiz = [sum(transposee[x]) for x in range(self.W)]

        Xmins, Xmaxs = [], []
        projection_precedante = 0
        for i, x in enumerate(project_horiz):
            if projection_precedante == 0 and x > 0:
                Xmins.append(i)
            elif projection_precedante > 0 and x == 0:
                Xmaxs.append(i-1)

            if i == (self.W - 1) and x > 0:
                Xmaxs.append(i)

            projection_precedante = x

        return [image(self.pixels[:, x_min:x_max + 1]).localisation() for x_min, x_max in zip(Xmins, Xmaxs)]



# fin class image


# ==============================================================================
#  Fonction de lecture des fichiers contenant les images modeles
#  Les differentes images sont mises dans une liste
# l'element '0' de la liste de la liste correspond au chiffre 0,
# l'element '1' au chiffre 1, etc.
# ==============================================================================

def lect_modeles():
    fichiers = ['_0.png', '_1.png', '_2.png', '_3.png', '_4.png', '_5.png', '_6.png',
                '_7.png', '_8.png', '_9.png']
    list_model = []
    for fichier in fichiers:
        model = image()
        model.load_image(fichier)
        list_model.append(model)
    return list_model


# ==============================================================================
# ==============================================================================

#   PROGRAMME PRINCIPAL

# ==============================================================================
# # Lecture image
# ==============================================================================

im = image()
im.load_image('test10.JPG')
im.display("image initiale")

# ==============================================================================
# Binarisation
# ==============================================================================
seuil = 145

binaris = im.binaris(seuil)
binaris.display("binaris")

#
# ==============================================================================
#  Localisation chiffre
# ==============================================================================
#

localisation = binaris.localisation()
localisation.display("localisation")

#
# ==============================================================================
# Test de la fonction resize
# ==============================================================================


resize_im = localisation.resize_im(100, 60)
resize_im.display("resize_im")

#
# ==============================================================================
# Test de la fonction similitude
# ==============================================================================

print("Similitudes :", "=============", sep="\n")
print("avec elle-même :", im.simil_im(im) * 100, "%")
print("avec son opposé :", im.simil_im(image(255 - im.pixels)) * 100, "%")
avec_nulle = image(np.zeros(im.pixels.shape, dtype=np.uint8)).simil_im(im.binaris(seuil)) * 100
print("avec la matrice nulle: {0:.2f} %\n".format(avec_nulle))

# ==============================================================================
# Lecture des chiffres modeles
# ==============================================================================

list_model = lect_modeles()
# test verifiant la bonne lecture de l'un des modeles, par exemple le modele '8'
list_model[8].display("modele 8")

# ==============================================================================
# Mesure de similitude entre l'image et les modeles
# et recherche de la meilleure similitude
# ==============================================================================


def assertion(a, b):
    if a == b:
        print("{} == {}: OK!".format(a, b))
    else:
        print("{} == {}: ERREUR!".format(a, b), file=sys.stderr)


def predire(image:image, models, seuil_blanc):
    image_traitee = image.binaris(seuil_blanc).localisation()
    return int(np.argmax([
                         image_traitee.resize_im(*model.pixels.shape)
                                      .binaris(240)
                                      .simil_im(model)
                         for model in models
                     ], axis=0))


prediction = predire(im, list_model, seuil)
assertion(prediction, 6)

for fichier, chiffre in [('test1.JPG', 4),
                         ('test2.JPG', 1),
                         ('test3.JPG', 2),
                         ('test4.JPG', 2),
                         ('test5.JPG', 2),
                         ('test6.JPG', 4),
                         ('test7.JPG', 7),
                         ('test10.jpg', 6),
                        ]:
    test = image()
    test.load_image(fichier, ecrire_nom=False)
    predict = predire(test, list_model, seuil)
    assertion(predict, chiffre)


def prediction_ligne(image:image, models, seuil_blanc):
    chiffres = []
    for chiffre in image.binaris(seuil_blanc).localisation_ligne():
        chiffre.display("")
        chiffres.append(str(predire(chiffre, models, seuil_blanc)))
    return " ".join(chiffres)


for fichier, chaine in [
    ('test8.JPG', '1 3 5 2'),
    ('test9.JPG', '1 8 4 5 6'),
]:
    test = image()
    test.load_image(fichier, ecrire_nom=False)
    prediction = prediction_ligne(test, list_model, 150)
    assertion(prediction, chaine)

