# Reaper Tool for Onset Extraction


This document explains how I set up an onset extraction tool using Reaper in the context of a research project on desynchronization analysis.

## Abstract

For more in-depth analyses, it was necessary to extract the onsets of each musician during experiments, with 25 microphones for each instrument or part of the instruments.

When extracting onsets, we encounter two major problems:

*   The microphones also capture ambient sounds, including those from other instruments. For example, the saxophone being close to the bass drum, the sound of it interferes with the sound of our saxophone. Therefore, it is necessary to clean the audio spectrally.
*   Detecting the onset is very particular because the transients differ for each type of sound produced. After cleaning the spectrum, it is necessary to manually determine where one can consider that a sound is triggered.

_Why Reaper?_

Originally, Thomas Wolf's method used two tools: Audacity to clean frequencies with a band EQ, and Sonic Visualizer to detect transients (using 2 algorithms) and export markers. One of the algorithms used resulted in a loss of temporal precision.

I propose to unify into a single tool. I chose Reaper because it is a DAW that I am familiar with and has a quite exhaustive and well-documented API for creating custom scripts and audio plugins. Moreover, it can be used without the need for a license. [Download Reaper](https://www.reaper.fm/)

## Method

> 💡 To take full advantage of all Reaper's features, it is strongly recommended to install these two plugins: [ReaPack](https://reapack.com/) and [SWS](https://www.sws-extension.org/). If you are new to Reaper, I highly recommend the tutorial videos by [Kenny Gioia](https://www.youtube.com/@REAPERMania/videos), the guru of Reaper.

The method is therefore based on 3 points: cleaning undesirable frequencies, detecting onsets, and exporting markers. Start by importing the desired media into a track, ensuring that the project's sample rate matches the files' (i.e., 48 kHz).

1.  **Spectral Cleaning**
    
    *   Reaper introduces a formidable tool for spectral manipulation on a spectrogram. We will use this tool to take the spectral profile of the noise to clean our track. Right-click on the item -> Spectral Edit.
    *   To display the spectrogram: right-click on the item -> Spectral Edits -> Always show Spectrogram, alternatively: View -> Peak Display Settings, select "spectrogram + peaks".
    *   To add an editing window: make a time selection of the area where you want to take the noise profile, right-click on the item -> Spectral Edits -> Add spectral edits to item, then manipulate the window. Detailed information is available in [section 7.39](https://dlz.reaper.fm/userguide/ReaperUserGuide712c.pdf).
    *   _Tips for selecting the noise profile_: given that tempos differ, there will come a time when the sound we want to keep is isolated from parasite sounds on the spectrogram and waveform. It can be clearly distinguished visually and by listening. Also choose an FFT size that suits the precision of your profile.
    *   Right-click on the window -> Solo spectral edit. You will then only hear the undesirable noise. Then add an FX to the track: View -> FX Browser, search for ReaFir (Reaper's built-in plugin) a dynamic FFT with non-linear phase EQ; and drag it onto the track.
    *   Manipulating the plugin is relatively simple, and additional information can be found in [section 16.12](https://dlz.reaper.fm/userguide/ReaperUserGuide712c.pdf). Adjust the FFT size to the same value as previously, choose the subtract mode, play your time selection (in loop) and check Automatically build noise profile (enable during noise). The spectral profile of the noise is compensated and in about 3 seconds it should disappear. It is also useful to manipulate the Gain to compensate for the volume loss due to spectral cleaning.
    *   Also you cn use other function of plugin, in particular you can enhance the selected sprecrum print using EQ and Compressor (hold Ctrl and adjust FFT) increase compressor ratio will reduce attack
    *   Of course, it is possible to add other plugins, such as a graphic EQ (ReaEQ), to enhance the frequencies to be preserved.
    *   Right-click on the window -> uncheck Solo spectral edit, right-click on the track -> Render/freeze tracks -> Freeze tracks to mono, to render the entire track. This operation is reversible.
2.  **Transient Detection**
    
    *   Transient detection is done using a built-in algorithm of Reaper. Right-click on the item -> item processing -> dynamic split item, check at transient and set Min slice length to the minimum duration between each transient (to know this value, make a time selection in the arrangement, in the transport bar you will see the length of your selection). More info on this tool in [section 7.36](https://dlz.reaper.fm/userguide/ReaperUserGuide712c.pdf). Select Action to perform: Add transient guide markers to selected items, to simply generate guides for each transient.
    *   Click on Set transient sensitivity to adjust the sensitivity and threshold of detection. Dotted guides will appear to preview the transients.
    *   Once you are satisfied with the segmentation, you can click on generate guide. They will be visible on the waveform.
3.  **Marker Export**
    
    *   Place the cursor at the beginning of the item, each press of TAB will move from one transient to another. At each transient, press M to place a marker, adjust the marker's position according to preference.
    *   For better visibility, remove the transients. Moreover, moving a transient transforms it into a stretch marker, which we do not want.
    *   Export the markers in the correct format with all the required time precision. Third-party scripts are necessary as there is a precision loss with the basic one. View -> Region/Marker Manager -> check Marker, select all markers, right-click -> export... Another method is to download the SWS script bundle which includes a marker list in which data can be exported with formatting to the millisecond.

## The Tool

The developed tool incorporates the three parts seen previously to accelerate the work. It requires the plugins ReaPack (package manager) and SWS (API extension), but also Ultraschall (another API extension) and Lokasenna GUI (a set of classes for the user interface). Upon starting the script, these last two should install automatically.

After installing the first two plugins, go to Extensions -> ReaPack -> Manage Repository -> Options -> Install new packages when synchronizing -> apply. The SWS packages should install ([ReaPack User Guide](https://reapack.com/user-guide)).

### Installation

1.  Place the folder _Analyse Dsync Tools_ into the Reaper Resource path (Options -> Show REAPER resource path in explorer/finder), Scripts folder.
2.  Actions -> Show action list or press "?", click on New Action -> Load Reascript and select the file "analyse-dsync - Onset Extraction Tool.lua".
3.  This customized action now appears in the list of Reaper actions. Press Run to start it. The missing packages should install.

### Overview

First, make sure to import your items into separate tracks. For this, drag and drop a batch of files towards Reaper. A dialog box will open asking you what action to perform.

The interface is presented as follows:

The header of the interface shows which track is selected with the name of the file in the track. The selected track is automatically soloed so you can freely listen to it individually. Pressing the arrows will move you from one track to another. "Estimated spacing in ms" is an approximate measure of the distance between each onset. To be calculated, at least 3 markers are required.

The frame just below presents 3 tabs representing the 3 stages of the method: cleaning, detection, and export. Each tab presents a frame of options (on the left) in which you can open Reaper's adjustment dialog boxes.

### Spectral Cleaning

Make sure your item is well-selected. "Get sound print" will add a spectral edit window to your item and add the VSTs ReaFIR for spectral cleaning and ReaEQ for a graphic equalizer to the track.
You can adjust knob to tweak bounded frequencies, frenquecy range, frequency/time contrast and compression.
Open ReaFIR, play the noise profile and check "Automatically build noise profile (enable during noise)". Uncheck when done. Use substract to remove desired frequencies, Compressor to enhance and EQ to increase or diminuss, play with the output gain and the spectrogram : hold Ctrl to move spectrum and Shift to draw on spectrum
You can then make adjustments on the desired frequencies with ReaEQ.

Once finished, press "done". The track will be frozen and the item will be locked. Obviously, you can reapply as much spectral cleaning as desired if you unlock the item.

### Transient Picking

The behavior of this part is to first generate Transient Guides and then convert them into Take Markers, markers with a position relative to the item.

Move one of the three sliders and the Transient Guides should be generated on your item. However, this is not very precise and rather gives an approximation of the settings to adopt. Therefore, I invite you to specify the generation of these guides by opening "dynamic split" and "transient detect". In "dynamic split", you can select the preset "analyse\_dsync\_split\_preset" and generate the guides in the most precise way possible. This method provides very good results if the parameters are correctly set.

Check "Update Transients" to automatically convert the Transient Guides into Take Markers. You will then be able to navigate between the Take Markers by clicking on the arrows, so you can individually listen to each onset, observe the waveform and manually adjust the position, or delete a marker by holding down Alt/Option. Finally, if you place the cursor of the arranger and press on the diamond, you create a new marker.

### Export

"Export selected item" will export the list of marker positions (onset, transient) in CSV format to the desired location to the millisecond.

"Export all items" will do so for all your tracks.

> 💡 Recommended settings:
> 
> *   Peak Settings: freq log = 2.9, curve = 2.0, contrast = 1.20, bright = 0... -> Scale peaks by square root (half of range is 12dB rather than 6dB)., rectify peak is also interesting
> *   Uncheck "Use zero crossings" (less precise but prevents clicks), check "Display threshold in media items while this window is open".


## Most recent edits

* I modify the FX chain to include 2 ReaFir in parallel, one for the Equalizer and the other for substracting, you can use both or one and change settings as you want obviously
* some minor bug to correct (script crash sometimes when adding spectral edit, markers doesn't update automaticly until changing track focus )