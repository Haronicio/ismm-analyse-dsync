# Reaper Tool for Onset Extraction

Ce document explique comment j'ai mis en place un outil d'extraction d'onset avec Reaper dans le cadre du projet de recherche sur l'analyse de la désynchronisation  

## Abstract

Pour des analyses plus poussé, il  a fallu extraire les onsets de chaque musicien lors des expérimentations, il y avait 19 micros(*) pour chaque instrument/ partie des instruments

Lors de l'extraction d'onsets on se heurte à deu problèmes majeurs :

* les micros captent aussi les bruits ambient dont les autres instruments, il en résulte que par exemple le sax êtant prêt de la grosse caisse(*), le bruit de celle-ci vient parasyter le son de notre sax, il faut donc nettoyer spectralement l'audio
* détécté l'onset est très particulier car les transients sont différents  pour chaque type de son produit, après avoir nettoyer le spectre il faut manuellement déterminer ou es ce qu'on peu considérer qu'un son ce déclenche

*Pouquoi Reaper ?*

Originalement la méthode de Thomas Wolf utilise deux outils, Audacity pour nettoyer les fréquences avec un EQ à bande et Sonic Vizualizer pour détetcter les transients (à l'aide de 2 algorithmes) et exporter les marqueurs, d'ailleurs l'un des algo utilisé fait perdre de la précision temporelle.

Je propose d'unifier en un seul et même outil, j'ai chois Reaper car c'est un DAW que je connais bien et qui a une API exaustive et largement documenté pour créer des Scripts et Plug in audio personnalisé

## Method

la méthode ce base sur donc 3 points, nettoyer les fréquences indésirable et détecter les onset créer/exporter les marqueurs , d'abord apporter le media dans une track, s'assurer que les sample rate du projet soit celui des fichiers son (càd 48 kHz)

1. **Nettoyage spectrale**
   * Reaper introduit un outil formidable de manipulation spectrale sur un spectrogramme, c'est cet outil que nous allons utiliser pour prendre le profile spectrale du bruit pour nettoyer notre piste, click droit sur l'item -> Spectral Edit
   * Afficher le spectrogramme : click droit sur l'item -> Spectral Edits -> Always show Spectrogramm , alternativement : View -> Peak Display Settings selectionne spectrogram + peaks
   * Ajouter une fenêtre d'édition : Faite une sélection de l'endroit ou l'on souhaite prendre le profile click droit sur l'item -> Spectral Edits -> Add spectral edits to item , ensuite manipuler la fenêtre, des information détailler ce trouve dans [la section 7.39]([https://](https://dlz.reaper.fm/userguide/ReaperUserGuide712c.pdf))
   * *tips pour sélectionner le profile du bruit* : êtant donné que les tempis sont différents, il arrivera un moment ou le son que l'on souhaite conserver ce retrouve isoler des sons parasyte sur le spectrogram et la waveform, on peut clairement le distinquer visuellement et à l'écoute, ce le meilleur profile a prendre, choississez aussi un FFT size assez grand pour être précis sur ce qu'on sélectionne
   * Click droit sur la fenêtre -> Solo spectral edit, vous n'entendrez alors que le bruit indésirable, ajouter alors un FX au track : View -> Fx Browser, chercher ReaFir (plugin built-in de Reaper) une FFT dynamic avec EQ à phase non linéaire; et glisse le sur le track
   * Manipuler le plugin est relativement siple et des information complémentaire peuvent être trouvé dans [la section 16.12](https://dlz.reaper.fm/userguide/ReaperUserGuide712c.pdf), ajuster FFT size à la même valeur que précedemment, choississez le mode substract, joué votre time selction (en boucle) et check Automatically build noise profile (enable during noise), le profile spectrale du bruit est compensé et en ~5 secondes il devrait disparaitre
   * Click droit sur la fenêtre -> uncheck Solo spectral edit, click droit sur l'item -> Glue items, pour rendre un nouvel item sans bruit (non destructif), le fx peut être désactivé

2. **Détection de transient**
   * La detection de transient ce fait à l'aide d'un algorithme built-in de reaper, click droit sur l'item -> item processing -> dynamic split item, check at transient et régle Min slice length Min silence length a des valeurs raisonnable pour l'item que vous souhaitez sélectionner, plus d'info sur cette outils dans [la section 7.36](https://dlz.reaper.fm/userguide/ReaperUserGuide712c.pdf), sélectionne Action to perform
   : Add transient guide markers to selected items, pour siplement générer des guides pour chaque transients
   *  click sur Set transient sensitivity pour paramétrer la sensibililté et le treshold de la détection, vous pouvez vous aider des guides qui aparaissent sur la waveform
   *  Une fois que vous êtes satisfait du découpage, vous pouvez clicker sur generate guide, ils seront visible sur la waveform

3. **Exportation des markers**
   *  Place le curseur au début de l'item, chaque appui sur TAB va passer d'un transient à l'autre, à chaque transient appuyer sur M pour placer un marker, ajuster la position du marker en fonction de la préférence
   *  Pour une meilleur visibilité enlever les transients , de plus, bouger un transient le transforme en stretch Marker, ce que nous ne voulons pas
   *  Exporter les marker au bon formattage avec toutes la précisions du temps requiet des script tiers car il y a une perte de précision avec celui de base, View -> Region/Marker Manager -> check Marker , select all Marker click droit -> export ... , l'autre moyen est de télécharger l'esemble de script SWS qui inclu un marker List dans lequel on peut exporter les données avec un formattage à la ms prêt

## TODO

Récupérer le range d'une fenêtre specrtral edit
Solo la fenêtre le profile du bruit
Appliquer Fir à partir de l'empreinte et glue item
Transient guide to Marker/Take Marker
Utiliser des Take Marker (locale a un item) plutôt que des Marker qui sont globale au projet
Export transient guide avec formattage