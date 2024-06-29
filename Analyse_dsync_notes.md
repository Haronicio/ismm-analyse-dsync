# Analyse Dsync Notes

Quelques notes sur l'état d'avancement du projet d'analyse de la désynchronisation de l'improvisation musicale

## Bref résumé des 3 derniers mois

Durant ces 3 derniers mois j'ai d'abord fait l'état de l'art du sujet, receuillant différent papiers sur l'analyse du mouvement de la synchronie et du traitement de signal.

[Focus of attention affects togetherness experiences and body interactivity in piano duos. - Bishop](https://doi.apa.org/doi/10.1037/aca0000555)
[COMPARING INERTIAL MOTION SENSORS FOR CAPTURING HUMAN MICROMOTION. - Burnim and al.](https://www.duo.uio.no/bitstream/handle/10852/106232/1/14-20_Riaz_et_a_SMC2023_proceedings.pdf)
[STUDIES ON THE RELATIONSHIP BETWEEN GESTURE AND SOUND IN MUSICAL PERFORMANCE](https://baptistecaramiaux.com/pdfs/caramiaux2012phd.pdf)
[Do Jazz Improvizers Really Interact? The Score Effect in Collective Jazz Improvisation](https://www.researchgate.net/publication/319457085_Do_jazz_improvisers_really_interact_The_score_effect_in_collective_jazz_improvisation)
[Body motion of choral singers](http://carloscancinochacon.com/laura/D'Amario-Ternstrom-et-al-2023.pdf)

La pluspart de la documentation scientifique pour les algos python viennent des études de EEG et ECG ainsi que l'analyse de la syncronie boursière

Ensuite j'ai élaboré une méthode pour récupérer un ensemble d'onset instrumental à partir de donnée audio, avec Reaper 

Un ensemble de statistique étudiant l'intensité de mouvemnt chez les participant en fonction des différents paramètres des expériences pour déssiner des tendances générales

Ellaboration de différente problématique hypothèse et axe de recherche basé sur les statistiques précédentes et du parcours des informations à dispositions (sons vidéos signaux)

## 

L'objectif actuel est donc de créer un outil (ensemble de fonctionnalité) d'analyse de syncronie entre différente série temporel, en l'occurance un signal (mouvement, position ,accélération, ...) et des onsets (instant dans le temps)

à partir de ces différents signaux il est  donc possible d'extraire des métriques qui nous aideront à caractérisé des caractéristiques individuels (limite de l'expérience), globale (éstimé)

Les problématique à éxplorer pour répondre a notre problème principale sont : 
Comment caractériser le mouvement des participants (oscillaire ou instrumentale) ? Les types de mouvements prédominant ? Les facteurs qui intéragissent directement avec le mouvement ? Comment le mouvement des différents sujets intéragissent entre eux ?

Ces problématique sont plus large que ce que les expérience ciblait à savoir, quelles sont les attracteurs de mouvement (autre qu'audio) :
* Clcik Tempo : cas de "base"
* Mask Attack :  visuelle
* Change : l'Attention
* 
la dernière problématique (un peu plus obscure) est comment née la sensation de groove  qui est de la science purement cognitive et difficile à mesurer

Les 3  analyses abordé sont les suivantes : une analyse qualitative ... , une analyse statistique : distribution entre catégories , une analyse axé traitement de signal (rétrocatif avec l'axe précédent) pour extraire les métriques 

### Première classification naive

Il me fallait un échantillon représentatif non influencé par des paramètres (complexité et facteur spécifique à l'expé) j'ai donc dans un premier temps pris Click Tempo qui est l'expérience  de base car les sujets semble être le moint influencé par les conditions( en particulier Same No qui contient 48 données de signaux), ensuite Click Tempo et Mask Attack pour les condition low No et high No, avec ces paramètres ces deux expériences reviennent au même, impliquant la polyrythmie et en discriminant la prépondérance de la jambe (click tempo) du bras (mask attack) et de la tête (les deux)

La classification se repose naivement sur 3 facteurs binaire (haut bas) qui sont : le signal est il sensiblement altéré par le bruit ? le signal est il sensiblement altéré par une grande variation d'amplitude ? Le signal tend il à avoir une fréquence ... ?

ces 3  paramètres sont déterminer visuellement et en étudiant la variance et les moyennes des métriques suivantes :
* Signal to Noise Ratio (rapport de bruit)
* Peak to Valley Ratio (rapport d'amplitude)
* Frequency Variation Ratio (S'il est élevé le signal est inconcistant en fréquence dans le temps)
* Amplitudes et Fréquences Maximale et Dominantes

j'ai pu alors attribuer une note sur 3 à chaque signaux : 
1. le signal diverge trop d'un cas global et est trop altéré
2. le signal est correcte, légèrement altéré mais exploitable
3. le signale est intéressant, uniforme et facilement exploitable

Cette classification (purement qualitative) peut déjà donné des tendnaces cependant le caractère subjectif de l'analyse et pas approfondie ne peux en l'état actuel  être utilisé a des find d'interprétation, la piste de ces ratios est a exploré

### Onsets

Des onsets, il y a une métrique intéressante qui est la déviation par rapport au tempo moyen calculé, celui-ci est calculé en faisant la moyenne sur l'intervalle de 8 offsets (certains musiciens jouant à la double croche), la déviation est donc la différence entre le tempo moyen et le tempo d'un paquet d'offset. Cette métrique semble indiqué simplement avec les onset sonores/instrumentaux si les sujets tentent de compenser un quelquonque retard dans leurs jeux à chaque instant
Cependant cette métrique est très variable, enfaite il se trouve que ca dépend aussi fortement de la précision du placement des onsets.

Il est possible de faire une petite étude comparative entre les déviations des mes onsets, de ceux de thomas et d'onset placer à l'oreille pour obtenir une estimation de cette précision

(ici plot distribution tempo deviation par musicien avec ci et std pour chaque musicien (Same No))
(ici quelques courbe de deviation en fonction du temps pour un musicien et un trial)

### Le traitement de signal 

Dans cette partie j'ai trouvé des métriques intéressante pour évaluer la syncronie des mouvements d'un muscien avec lui même et ses onsets extraits.

il y a l'approche statique(sur l'ensemble des deux signaux) et l'approche dynamique(sur des segments/fenêtre)

voici comment sont traité en amont les signaux :
le signal de mouvement est centré pour retirer la composante DC, puis filtré entre 0.2 Hz et 8.5 Hz avec un band-pass : en effet le tempo varie de 60 à 240 bpm 
donc de 1 Hz à 4 Hz, si le mouvement est à la ronde 0.25 Hz à 1 Hz, si le mouvement est à la double croche 8 Hz à 32 Hz mais il est très peu probale d'un point de vue physionomique qu'un mouvement soit aussi rapide même avec des amplitudes très petites. Concernant les onsets, le signal est à 1 lorsqu'il y a un onset et 0 sinon, ensuite ce signal est approximé par une fonction de fenêtre à la largeur du tempo moyen (ATTENTION CHEVAUCHEMENT), j'en ai essayé plusieurs , et la fenêtre de taylor (ici fenêtre de taylor avec freq) semble être la plus intéressante, elle ajoute un minimum de distorsion pour une bonne résolution fréquentiel avec des harmoniques qui on une magnitude faible, bien que ce soit une aproximation, l'apodization permet d'éxagéré l'extraction des features des signaux, le signal est ensuite centré

Tout d'abord il y a l'auto-corrélation et la cross-corrélation signal/onset , elles permettent d'extraire la periodicité d'un signal ou la periodicité de l'un par rapport à l'autre, il y a trois possibilité :
* l'autocorrélation est parfaitement superposé à la corrélation : le mouvment est probanlement un geste instrumentale (dépend de la partie du corps) ou accompagne le geste instrumentale
* l'autocorrélation est la corrélation avec une periode en plus : le mouvement est décomposé en plusieurs étape, donc le mouvement est à une subdivision rthmique inférieur au tempo ciblé
* la corrélation est a l'autocorrélation avec une periode en plus : bizarrement ce cas est plus rare, celui ci suggère que la subdivision du mouvement est plus grande au tempo joué

il ya ensuite la DTW (dynamic time warping) qui permet de comparer les segments des signaux entre eux et ainsi déterminer s'il y en a un plus rapude que l'autre, il en résulte  d'une pente et d'un taux de superposition, si la pente est > 1 les onsets sont plus rapide que le signal, sinon le signal est plus rapide que les onsets, généralement il est très proche de 1, le taux de superposition avalue ensuite à quelle point les segments sont vraiment superposé pour toute la longeur des signaux (avec une tolérance de 100 samples de décalage) c'est à dire que si ce taux est proche de 100% , les onset sont parfaitement en même temp que les signaux, s'il est autour de 50% le mouvement et les onsets essayent de se rattraper

La cohérence sur l'ensemble du signal et des onsets extrait les fréquences cohérentes entre elles, on peut aussi obtenir les amplitudes et les fréquences globales caractérisant le jeu (qu'on a déjà avec le tempo moyen et le metronome) ainsi que le mouvement avec les FFT des ondes correspondants.

pour l'approche dynamique il ya plusieurs métrique très similaire : la cohérence dans le temps, les spectrogrammes, la cross corrélation dans le temps, mais aussi Instantaneous phase synchronie (+ transformé de Hilbert) et la corrélation optimized warping

il sera ensuite utile d'effectuer une causalité de granger sur ces arrays entre 2 musciens d'un même trial pour estimer s'il y a une causalité dans les variations : une attraction

### les métriques pour les statistiques

Pour le moment les métriques statiques retenu sont les suivantes :

* cross correlation en 0
* coherence maximum
* fréquence assoscié à la coherence maximum
* dephasage pour maximiser la cross correlation en 0
* cross correlation en 0 avec le déphasage
* intensité moyenne
* std de l'intensité
* ci68 de l'intensité
* tempo moyen
* déviation moyenne par rapport au tempo moyen
* std de la déviation moyenne par rapport au tempo moyen
* ci68 de la déviation moyenne par rapport au tempo moyen
* slope du  dtw
* % alignement du dtw

Pour chaque couple de métrique il est utile de regarder la linéarité entre elles avec la corrélation de Pearson et aisni tracer les moyennes et les distributions pour avoir des tendances plus fiable

### SyncPy 

J'ai commencé à corrigé syncpy pour l'adapter à l'environnement conda de notre projet, une fois que toutes ces métriques seront regroupé, on pourra étudier la synchronie de l'ensemble de ces métriques  
