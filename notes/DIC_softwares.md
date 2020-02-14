# Digital Image Correlation softwares and codes

## Open Source

- **DICe**, from Dan Turner at Sandia.
  ([https://github.com/dicengine/dice](https://github.com/dicengine/dice))
  
  *Both local and global DIC algorithms (global yet to be released) [ref](http://dicengine.github.io/dice/)*

- **OpenDIC** code developed by Nicolas Vanderesse at ETS, montreal.
  ([Nicolas.Vanderesse@ens.etsmtl.ca](mailto:Nicolas.Vanderesse@ens.etsmtl.ca))
  
  *code non accessible*

- custom Matlab-based code, by Christoph Eberl - Daniel Gianola (UCSB)
  https://www.mathworks.com/matlabcentral/fileexchange/12413-digital-image-correlation-and-tracking
  
  *use `cpcorr` from Matlab's image processing toolbox  [ref](https://fr.mathworks.com/help/images/ref/cpcorr.html): "only moves the position of a control point by up to four pixels. Adjusted coordinates are accurate up to one-tenth of a pixel. `cpcorr` is designed to get subpixel accuracy from the image content and coarse control point selection."*

- **Correla** (Valéry Valle, Pprime)
  https://www.pprime.fr/?q=en/photomechanics
  
  *code non directement accessible*

- **YADICS** (Yet Another DIC software), LML - Lille
  [http://yadics.univ-lille1.fr/wordpress/](http://yadics.univ-lille1.fr/wordpress/)
  
  *le code semble non maintenu, date de 2015, compilation problématique*

- **Ncorr** (Georgia Tech)
  [http://www.ncorr.com/](http://www.ncorr.com/)
  Ncorr is an open source 2D digital image correlation MATLAB program.

- **pyDIC** (Damien André, Centre Européen de la Céramique, Limoges)
  [https://gitlab.com/damien.andre/pydic](https://gitlab.com/damien.andre/pydic)
  *python + OpenCV: algorithme basique (use cv2.calcOpticalFlowPyrLK)*

- **elastiX**
  http://elastix.isi.uu.nl/
  
  *Medical image registration problems, based on [Insight Segmentation and Registration Toolkit](http://www.itk.org) (ITK). Accompanied by the API [SimpleElastix](http://simpleelastix.github.io/), making it available in languages like C++, Python, Java, R, Ruby, C# and Lua.*

- **pyxel**
  https://github.com/jcpassieux/pyxel
  *python library for experimental mechanics using finite elements* 
  *Jean-Emmanuel Pierré, [Jean-Charles Passieux](http://institut-clement-ader.org/author/jcpassieux/), Jean-Noël Périé (ICA - Institut Clément Ader, INSA Toulouse)* **FE based**
  use Scipy `RectBivariateSpline` [line 1059](https://github.com/jcpassieux/pyxel/blob/77a3dc433958e6541772e4ab200cf8e0ac9c7b10/pyxel.py#L1059) for image interpolation

- **µDIC**
  https://github.com/PolymerGuy/muDIC
  Digital Image Correlation in Python

- **py2DIC**
  http://github.com/Geod-Geom/py2DIC/
  *Geodesy and Geomatics Division of the Sapienza University ofRome*
  uses `cv2.matchTemplate`  [doc](https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template_matching/template_matching.html)

- **dolfin_dic**
  https://bitbucket.org/mgenet/dolfin_dic/src/master/
  https://www.sciencedirect.com/science/article/abs/pii/S1361841518305346

## Commercial package

- **Vic2D** from Correlated Solutions

- **Aramis (GOM)**
  [https://www.gom.com/fr/systemes-de-mesure/aramis.html](https://www.gom.com/fr/systemes-de-mesure/aramis.html)

- **Correli**
  Le logiciel de corrélation d’image CORRELI STC est développé par HOLO3, en partenariat avec le LMT de Cachan et Airbus Group Innovations.[https://www.correli-stc.com/accueil.html](https://www.correli-stc.com/accueil.html)
  
  *ENS Cachan, François HILD, Stéphane ROUX
  code pas en ligne (apparement Matlab)*

- **EikoSim**
  https://eikosim.com/entreprise/
  *De par son expertise et son lien étroit avec le laboratoire de recherche LMT (ENS Paris-Saclay), EikoSim...*
