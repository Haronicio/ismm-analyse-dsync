-- Script Name: Onset Extraction Tool

--[[
    Author : Haron DAUVET-DIAKHATE (haron.dauvet@live.com)
    Company : IRCAM UMRSTMS ISMM
    Date : 24/04/2024
    Vesrion : 1.O
    Tested on : Windows 10, MacOS 11
    Min required Reaper vers. : 7.15
]]

-- Dict for All Command ID used in the tool (better organization)
command = { ["Spectrogram: Add spectral edit to item"] = 42302,
            ["View: Show peaks display settings"] = 42074,
            ["Transient detection sensitivity/threshold"] = 41208,
            ["Edit: Dynamic split items"] = 40760,
            ["Clear transient guides"] = 42027,
            ["Item: Quick add take marker at play position or edit cursor"] = 42390,
            ["Track: Freeze to mono"] = 40901,
            ["Calculate transient guides"] = 42028,
            ["SWS: Zoom to selected items, minimize others"] = reaper.NamedCommandLookup('_SWS_ITEMZOOMMIN') }

DOC_URL = "https://github.com/Haronicio/ismm-analyse-dsync/tree/main/Reaper"

debug = {}


-- Global variable for the current context

-- int
TrackNB = 0

-- int 
TransientNB = 0

-- int
Selected_Track = 0

-- int
Selected_Transient = 1

-- Media Item
Selected_Item = nil


-- Tab of onset tab 1..TransientNB (for each track 0..TrackNB-1)
Onsets_Item = {}


-- ref to actual trak raw transint guides

Selected_TransientsGuide = {}

--

Selected_SampleRate = 0



-- Video

Video_Enable = false
Video_Track = 0


--tmp
function Print_console(str)
    reaper.ShowConsoleMsg(str)
end


function main()
  
    -- SWS and Reapack Plugin tets
    if not reaper.APIExists("ReaPack_BrowsePackages") then
        reaper.MB("Please install SWS/S&M EXTENSION following this link \nhttps://www.sws-extension.org/", "Missing SWS/S&M EXTENSION", 0)
        return
    end
    if not reaper.APIExists("CF_GetSWSVersion") then
        reaper.MB("Please install ReaPack: Package manager for REAPER following this link \nhttps://reapack.com/", "Missing ReaPack", 0)
        return
    end

    loadInstallUltraschallAPI()
    -- ultraschall.ApiTest()
    -- reaper.ShowConsoleMsg(GetThisScriptFolder().."\n")

    -- Should begin with that
    Selected_Item = SelectFirstMediaItemOnSelected_Track()
    SoloTrack(Selected_Track)
    UpdateTransient_Selected_Item()

    -- Print_console(reaper.BR_GetMediaTrackFreezeCount(reaper.GetTrack(0,Selected_Track)))

    -- Previous_item()
    -- ConvertTransientGuidesToTakeMarkers()
    -- UpdateTransient_Selected_Item()
    -- Next_item()
    -- ConvertTransientGuidesToTakeMarkers()
    -- UpdateTransient_Selected_Item()
    
    -- ZoomIntoPosition(Onsets_Item[Selected_Track][Selected_Transient])

    -- exportTakeMarkersToCSV(Onsets_Item[Selected_Track])
    -- LoadReaFIR_FxChain()

    -- updateTransientCalculation(2,-60.0)
    --[[
    for i, onsets in pairs(Onsets_Item) do
            for j, take_mark in ipairs(onsets) do
                    Print_console("Item "..tostring(i).." :\tMarker "..tostring(j).." -> "..tostring(take_mark).."\n")
            end
    end
    ]]




    -- showPeakDisplaySetting()
    -- showDynamicSplitItems()
    -- showTransientDetectionSetting()
    
    -- AddSpectralEdit()
    -- ModifyFirstSpectralEdit()

    -- LoadReaFIR_FxChain()

    --[[
    sectionName = {}
        --reaper.Main_OnCommand(40760, 0) -- Item: Dynamic split items...
        
        
    window = reaper.JS_Window_Find('Media Explorer', true) 

        if window then
        for i = 0, 100000 do
                elements = reaper.JS_Window_FindChildByID(window, i)
                parameter = reaper.JS_Window_GetTitle(elements, "")
                
                if parameter ~= "" then
                    -- sectionName[i] = parameter
                    reaper.ShowConsoleMsg(tostring(i).." "..parameter.." \n")
                end
        end
        reaper.JS_Window_OnCommand(window, tostring(1011))
            
        end
        
        updateDynamicSplitPreset(65, 220)
        ]]


end

function AddSpectralEdit()
  -- Ensure there's at least one item selected
  local itemCount = reaper.CountSelectedMediaItems(0)
  if itemCount == 0 then
      reaper.ShowMessageBox("No items selected. Please select an item to add a spectral edit.", "Error", 0)
      return
  end

  -- Retrieve the first selected item
  local item = reaper.GetSelectedMediaItem(0, 0)
  
  -- Perform action Spectrogram: Add spectral edit to item
  reaper.Main_OnCommand(command["Spectrogram: Add spectral edit to item"], 0)

  
  reaper.UpdateArrange() -- Refresh the arrange view to reflect the changes
  
  --[[ Here we assume there's a specific action for adding a spectral edit.
  local actionCommandID = reaper.NamedCommandLookup("blabla")
  
  -- Perform the action if valid
  if actionCommandID ~= nil and actionCommandID ~= 0 then
      reaper.Main_OnCommand(actionCommandID, 0)
  else
      reaper.ShowMessageBox("Spectral edit action not found. Please check the action ID.", "Error", 0)
      return
  end
  ]]
  
  -- Example action to ensure replay for TCP is on.
  -- This is a placeholder, as the exact action may vary. Ensure to replace it with the correct command ID.
  -- reaper.Main_OnCommand(commandIDForReplayTCP, 0)

  -- Close script window/dialog (this action does not usually require a command as scripts typically terminate after execution)
end


-- Made my own but ultrascallAPI have a good behavior for that
function ModifyFirstSpectralEdit()
    -- Ensure there's at least one item selected
    local itemCount = reaper.CountSelectedMediaItems(0)
    if itemCount == 0 then
        reaper.ShowMessageBox("No items selected. Please select an item to modify a spectral edit.", "Error", 0)
        return
    end

    -- Get the first selected item and its active take
    local item = reaper.GetSelectedMediaItem(0, 0)
    local take = reaper.GetActiveTake(item)
    if not take then return end  -- Exit if there's no active take

    -- Get the current state chunk of the take
    local retval, takeChunk = reaper.GetItemStateChunk(item, "", false)
    if not retval then return end -- Exit if unable to get the state chunk

    -- Find and toggle the solo state and FFT size for the first spectral edit
    local modifiedChunk = ""
    local foundSE = false
    for line in takeChunk:gmatch("[^\r\n]+") do
        if line:find("SPECTRAL_EDIT ") and not foundSE then
            -- Found the first spectral edit, now toggle the bypass state
            foundSE = true

            local spectralEditValues = parseSpectralEdit(line)
            
            -- bypass at index 9, 0 no 1 bypass 2 solo
            spectralEditValues[9] = 2
            
            -- replace value to solo spectral edit
            local newline = "SPECTRAL_EDIT"
            for _,value in pairs(spectralEditValues) do
              newline = newline.." "..value
            end
            
            line = newline
            
        end
        -- Change spectral config if foundSE true (strange behavior but work as excepted)
        if line.find(line,"SPECTRAL_CONFIG ") and foundSE then
          line = line.."\n".."SPECTRAL_CONFIG " .. 8192
        end
        
        modifiedChunk = modifiedChunk .. line .. "\n"
    end

    -- Set the modified state chunk back to the take

    if foundSE then
        reaper.SetItemStateChunk(item, modifiedChunk, false)
        reaper.UpdateArrange() -- Refresh the arrange view to reflect the changes
    else
        reaper.ShowMessageBox("No spectral edit found to bypass.", "Info", 0)
    end
end

-- Function to parse the spectral edit line and return a table of values
function parseSpectralEdit(line)
    local pattern = "[+-]?%d+%.?%d*"
    local values = {}  -- Table to store the extracted values
    -- Iterate over each space-separated value in the line
    for value in line:gmatch(pattern) do
        -- Attempt to convert the string to a number, if possible
        local num = tonumber(value)
        if num then
            -- If it's a valid number, add to the table
            table.insert(values, num)
        else
            -- If not a number, add the string value (e.g., the "SPECTRAL_EDIT" label)
            table.insert(values, value)
        end
    end
    return values
end

-- 
function FreezeTrack(trackIndex)
    -- Get the track object by index; trackIndex is zero-based and does not include the master track
    local track = reaper.GetTrack(0, trackIndex)
    if track then
        -- Unselect all tracl and Set the track as selected
        reaper.SetOnlyTrackSelected(track)
        reaper.Main_OnCommand(command["Track: Freeze to mono"],0)
    end


end

-- Use LoadReaFIR_FxChain instead (can load FX parameters that not appear as fxindex
function addReaFIRToSelectedTrack_Item()
    -- Get the number of selected media items
    local itemCount = reaper.CountSelectedMediaItems(0)
    if itemCount == 0 then
        reaper.ShowMessageBox("No items selected. Please select an item.", "Error", 0)
        return
    end
    
    -- Loop through all selected items
    for i = 0, itemCount - 1 do
        -- Get the selected item
        local item = reaper.GetSelectedMediaItem(0, i)
        if item then
            -- Get the track associated with the item
            local track = reaper.GetMediaItem_Track(item)
            
            -- Add ReaFIR to the track's FX chain
            local fxIndex = reaper.TrackFX_AddByName(track, "ReaFir (FFT EQ+Dynamics Processor) (Cockos)", false, 1)
            
            if fxIndex == -1 then
                reaper.ShowMessageBox("Failed to add ReaFIR to the track.", "Error", 0)
                return 
            end
            reaper.ShowMessageBox("ReaFIR added successfully.", "Success", 0)
            
            -- Wait a bit for the plugin to be fully loaded (may not be necessary but here for safety)
            reaper.defer(function() end) 
            
            
            -- Get the number of parameters for the ReaFIR instance
            local paramCount = reaper.TrackFX_GetNumParams(track, fxIndex)
            
            -- Iterate through all parameters and print their names
            for paramIndex = 0, paramCount - 1 do
                local _, paramName = reaper.TrackFX_GetParamName(track, fxIndex, paramIndex, "")
                reaper.ShowConsoleMsg("Param " .. paramIndex .. ": " .. paramName .. "\n")
            end
        end
    end
    
end

-- this use ultrashall API
function LoadReaFIR_FxChain()
  -- get track
  local itemCount = reaper.CountSelectedMediaItems(0)
  if itemCount == 0 then
      reaper.ShowMessageBox("No items selected. Please select an item.", "Error", 0)
      return
  end
  local item = reaper.GetSelectedMediaItem(0, 0)
  if item then
  
      -- Get the track/ID associated with the item
      local track = reaper.GetMediaItem_Track(item)
      local trackID = reaper.CSurf_TrackToID(track, false)
      
      -- get the FX chain from file, 0 Track 1 Take
    --   local reaFIR_FXChain = ultraschall.LoadFXStateChunkFromRFXChainFile("Analyse.RfxChain",0)
      local reaFIR_FXChain = ultraschall.LoadFXStateChunkFromRFXChainFile(reaper.GetResourcePath().."/Scripts/Analyse Dsync Tools/Analyse.RfxChain",0)
      --reaper.ShowConsoleMsg(reaFIR_FXChain)
      
      -- get the state chunk of track
      local retval,trackStateChunk = reaper.GetTrackStateChunk(track,"",false)
      -- reaper.ShowConsoleMsg(trackStateChunk)
      -- reaper.ShowConsoleMsg(tostring(retval))
      
      -- set the FX chain into the chunk string
      local retval, alteredTrackStateChunk = ultraschall.SetFXStateChunk(trackStateChunk,reaFIR_FXChain)
      -- reaper.ShowConsoleMsg(alteredTrackStateChunk)
      -- reaper.ShowConsoleMsg(tostring(retval))
      
      -- set the new chunk state into the track chunk state
      local retval = reaper.SetTrackStateChunk(track,alteredTrackStateChunk,false)
      -- alternatively with ultraschallAPI
      -- retval = ultraschall.SetTrackStateChunk_Tracknumber(trackID, alteredTrackStateChunk)
      -- reaper.ShowConsoleMsg(tostring(retval))
      
      -- try to add build noise profile
      -- local fxChunk = ultraschall.Get
    --   reaper.ShowConsoleMsg(tostring(ultraschall.CountParmAlias_FXStateChunk(reaFIR_FXChain,1)))
      
      -- test
      local retval,trackStateChunk = reaper.GetTrackStateChunk(track,"",false)
      -- reaper.ShowConsoleMsg(trackStateChunk)
      -- reaper.ShowConsoleMsg(tostring(retval))
      
      -- update ARA
      reaper.UpdateArrange()
      
      
      --[[ to test if the file have a right syntax
      
      local filePath =  reaper.GetResourcePath().."\\exemple.txt"
      local file, err = io.open(filePath, "w")
      
      if not file then
          reaper.ShowConsoleMsg("Error opening file: " .. err)
      else
          file:write(alteredTrackStateChunk)
          reaper.ShowConsoleMsg("Done")
          
          file:close()
      end
      ]]
  end
end


-- alternatively use ultraschallAPI to get Scripts folder Path
function GetThisScriptFolder()
    local separator = ""
    local os = reaper.GetOS() 
    if os ~= "Win32" and os ~= "Win64" then
      separator = "/"
    else
      separator = "\\"
    end
    local script_path = reaper.GetResourcePath()
    return "\'"..script_path..separator.."Analyse Dsync Tools\\".."\'"
end

function CalculateTransientGuides()
    reaper.Main_OnCommand(command["Calculate transient guides"],0)
    Selected_SampleRate,Selected_TransientsGuide = getTransientMarkers(Selected_Item)
end

-- old version with StretchMarker
function ConvertStretchMarkerToTakeMarkers()

    local takeMarkers = {}

    -- Count the number of selected media items
    local itemCount = reaper.CountSelectedMediaItems(0)

    if itemCount == 0 then
        reaper.ShowMessageBox("No items selected", "Warning", 0)
        return
    end

    -- Iterate over all selected media items
    for i = 0, itemCount-1 do
        -- Get the media item
        local item = reaper.GetSelectedMediaItem(0, i)
        -- Get the number of takes in the item
        local takeCount = reaper.CountTakes(item)
        

        -- Iterate over all takes in the item
        for j = 0, takeCount-1 do
            -- Get the take
            local take = reaper.GetTake(item, j)
            
            if reaper.TakeIsMIDI(take) then
                -- Skip MIDI takes
                goto continue
            end

            -- Get the number of (transient guides) stretch marker in the take
            local transientCount = reaper.GetTakeNumStretchMarkers(take)
            -- reaper.ShowConsoleMsg(tostring(transientCount))

            -- Iterate over all transient guides and create take markers
            for k = 0, transientCount-1 do
                local _, pos = reaper.GetTakeStretchMarker(take, k)
                -- Add take marker at transient position, take , marker number (-1 auto), name, optional position, optional color
                local markerIndex = reaper.SetTakeMarker(take, -1,"", pos)
                
                -- save take marker into a table of take number,absolute time
                local itemStart = reaper.GetMediaItemInfo_Value(item, "D_POSITION")
                local absTime = itemStart + reaper.GetMediaItemTakeInfo_Value(take, "D_STARTOFFS") + pos
                --table.insert(takeMarkers, {takeNumber = markerIndex + 1, time = absTime})
                takeMarkers[markerIndex + 1] = absTime
            end

            ::continue::
        end
    end

    reaper.UpdateArrange()
    return takeMarkers
end

function ConvertTransientGuidesToTakeMarkers()
    local takeMarkers = {}

    -- Count the number of selected media items
    local itemCount = reaper.CountSelectedMediaItems(0)

    if itemCount == 0 then
        reaper.ShowMessageBox("No items selected", "Warning", 0)
        return
    end

    -- Get the take (hypothesis : only 1 take per item)
    local take = reaper.GetTake(Selected_Item, 0)

    -- Iterate over all transient guides and create take markers
    -- for k,pos in ipairs(getTransientMarkersTimeStamps(Selected_Item)) do
    -- slitly better performance
    for k,pos in ipairs(rawToTimestamps(Selected_SampleRate,Selected_TransientsGuide)) do
        -- Add take marker at transient position, take , marker number (-1 auto), name, optional position, optional color
        local markerIndex = reaper.SetTakeMarker(take, -1,"", pos,getTrackColorAndNegative(Selected_Track))
        takeMarkers[markerIndex + 1] = pos
    end

    reaper.UpdateArrange()
    return takeMarkers
end


-- Manage install and load UltrashallAPI to manage Chunk FX
function checkAPIAvailability()
    local status, err = pcall(dofile, ultraschallAPI_Path)
    if not status then
        -- The API is not yet available; print the error and retry after a short delay
        -- reaper.ShowConsoleMsg("Ultraschall API is not available yet: " .. tostring(err) .. "\n")
        reaper.defer(checkAPIAvailability) -- Retry after a short delay
    else
        -- The API is available; you can now safely call its functions
        -- reaper.ShowConsoleMsg("Ultraschall API is loaded.\n")
        ultraschall.ApiTest()
    end
end
function loadInstallUltraschallAPI()
  ultraschallAPI_Path = reaper.GetResourcePath().."/UserPlugins/ultraschall_api.lua"
  local status, err = pcall(function()dofile(ultraschallAPI_Path)end)
  if not status then
      print("An error occurred: " .. err)
      local ok,error = reaper.ReaPack_AddSetRepository("Ultraschall-API"
                                                    ,"https://raw.githubusercontent.com/Ultraschall/ultraschall-lua-api-for-reaper/master/ultraschall_api_index.xml"
                                                    , true
                                                    , 2)
      if ok then
        reaper.ReaPack_ProcessQueue(true)
        -- Waiting to API succesfully download
        checkAPIAvailability()
      end
  end
end

-- Function to convert project time (seconds) to absolute time (milliseconds)
function secondsToMilliseconds(seconds)
    return seconds * 1000
end

--update dynamic split preset alternatively change directly with other method
function updateDynamicSplitPreset(minSilenceLength, minSilenceLengthEnd)
    local resourcePath = reaper.GetResourcePath()
    local dynSplitIniPath = resourcePath .. "/reaper-dynsplit.ini"
    local fileContent = {}
    local lineFound = false

    -- verif

    local file = io.open(dynSplitIniPath, "w+")
    file:close()
    
    -- Read the current content and update if the line exists
    for line in io.lines(dynSplitIniPath) do
        if line:find("^analyse_dsync_split_preset") then
            line = "analyse_dsync_split_preset 1 " .. minSilenceLength .. " " .. minSilenceLengthEnd .. " -24 -6 98 2 50 0 0"
            lineFound = true
        end
        table.insert(fileContent, line)
    end
    
    -- If the line does not exist, append it
    if not lineFound then
        table.insert(fileContent, "analyse_dsync_split_preset 1 " .. minSilenceLength .. " " .. minSilenceLengthEnd .. " -22.38 -4.43 98 3 50 0 0")
    end
    
    -- Save the updated content back to the file
    local file, err = io.open(dynSplitIniPath, "w")
    if not file then
        reaper.ShowMessageBox("Error opening file for writing: "..err, "Error", 0)
        return false
    end
    
    for _, line in ipairs(fileContent) do
        file:write(line .. "\n")
    end
    
    file:close()
    return true
end

function updateDynamicSplit(minSilenceLength, minSilenceLengthEnd)
    -- if not (reaper.SNM_SetIntConfigVar("minslice",minSilenceLength) and reaper.SNM_SetIntConfigVar("minsilence",minSilenceLengthEnd)) then
    --     reaper.ShowMessageBox("Can't access to reaper.ini", "Error", 0)
    -- end
    local reaperIniPath = reaper.GetResourcePath() .. "/REAPER.ini"
    local fileContent = {}
    local found = false
    local searchKey = "minslice"
    local newValue = minSilenceLength

    -- Read the existing .ini file
    for line in io.lines(reaperIniPath) do
        if line:find("^" .. searchKey .. "=") then
            -- Modify the specific line with the new value
            table.insert(fileContent, searchKey .. "=" .. newValue)
            found = true
        else
            -- Keep other lines unchanged
            table.insert(fileContent, line)
        end
    end

    if not found then
        -- If the key wasn't found, add it
        table.insert(fileContent, searchKey .. "=" .. newValue)
    end

    -- Write the modified content back to the file
    local file = io.open(reaperIniPath, "w")
    if file then
        file:write(table.concat(fileContent, "\n"))
        file:close()

        reaper.UpdateArrange()

        return true, "The REAPER.ini file has been successfully updated."
    else
        return false, "Failed to open REAPER.ini for writing."
    end

    
end

function updateTransientCalculation(snsv,trsh)
    -- sensivitity 0.0 to 1.0 default 0.50
    -- treshold -60 to 0 default -17
    if not (reaper.SNM_SetDoubleConfigVar("transientsensitivity",snsv) and reaper.SNM_SetDoubleConfigVar("transientthreshold",trsh)) then
        reaper.ShowMessageBox("Can't access to reaper.ini", "Error", 0)
    end

    reaper.UpdateArrange()
end

function GetSoundPrint()
    AddSpectralEdit()
    ModifyFirstSpectralEdit()
    LoadReaFIR_FxChain()
end



-- Function to write take marker data to a CSV file
function exportTakeMarkersToCSV(takeMarkers,--[[optional]] filePath)

    -- Verify takeMarkers len
    if #takeMarkers == 0 then
          reaper.ShowMessageBox("No take markers found in selected items.", "Info", 0)
          return false
    end
    
    local retval = 0

    -- open file
    if not filePath then
       retval,filePath =reaper.JS_Dialog_BrowseForSaveFile("Save CSV", "", reaper.GetTakeName(reaper.GetActiveTake(Selected_Item))..".csv", "CSV files (*.csv)\0*.csv\0All Files (*.*)\0*.*\0")
    end
    
    
    local file, err = io.open(filePath, "w")
    if not file then
        reaper.ShowMessageBox("Error opening file for writing: " .. err, "Error", 0)
        return false
    end

    -- file:write("Take Number,Absolute Time (ms)\n")
    file:write("Absolute Time (s)\n")
    for _, markerData in ipairs(takeMarkers) do
        -- file:write(string.format("%d,%.3f\n", markerData.takeNumber, secondsToMilliseconds(markerData.time)))
        file:write(string.format("%f\n", markerData))
    end

    file:close()
    
    reaper.ShowMessageBox("Take markers exported successfully to:\n" .. filePath, "Success", 0)
    return true
end

function exportAll()
    -- Set the dialog caption and the initial folder (empty for default location)
    local caption = "Please select a folder"
    local initialFolder = ""  -- You can specify a default path here

    -- Call the function to open the browse folder dialog
    local retval, folderPath = reaper.JS_Dialog_BrowseForFolder(caption, initialFolder)

    -- Check if the user selected a folder
    if retval and folderPath ~= "" then
        reaper.ShowConsoleMsg("Selected folder: " .. folderPath .. "\n")
    else
        reaper.ShowConsoleMsg("No folder was selected.\n")
    end

    for index, takeMarkers in pairs(Onsets_Item) do
        -- Print_console(index)
        exportTakeMarkersToCSV(takeMarkers,folderPath.."/"..reaper.GetTakeName(reaper.GetActiveTake(reaper.GetTrackMediaItem(reaper.GetTrack(0, index), 0)))..".csv")
    end
end

function showPeakDisplaySetting()
   reaper.Main_OnCommand(command["View: Show peaks display settings"], 0)
end

function showTransientDetectionSetting()
  reaper.Main_OnCommand(command["Transient detection sensitivity/threshold"], 0)
end

function showDynamicSplitItems()
  reaper.Main_OnCommand(command["Edit: Dynamic split items"], 0)
end


-- current item/track management

-- return item selected
function SelectFirstMediaItemOnSelected_Track()
    -- Update the number of tracks in the current project (excluding master)
    TrackNB = reaper.CountTracks(0)

    -- Deselect all items to start fresh
    reaper.SelectAllMediaItems(0, false)

    -- Get the first media item on Selected_track track
    local item = reaper.GetTrackMediaItem(reaper.GetTrack(0,Selected_Track), 0)

    -- If an item was found
    if item then
        reaper.SetMediaItemSelected(item, true)
    end

    -- Focus this item/track and minimize other
    reaper.Main_OnCommand(command['SWS: Zoom to selected items, minimize others'],0)

    -- Update the arrangement view to reflect the item selection changes
    reaper.UpdateArrange()

    return item
end

function Previous_item()
    -- Update the number of tracks in the current project (excluding master)
    TrackNB = reaper.CountTracks(0)
    UnsoloTrack(Selected_Track)
    Selected_Track = (Selected_Track - 1) % TrackNB
    Selected_Item = SelectFirstMediaItemOnSelected_Track()
    UpdateTransient_Selected_Item()
    Selected_Transient = 0
    SoloTrack(Selected_Track)
end

function Next_item()
    -- Update the number of tracks in the current project (excluding master)
    TrackNB = reaper.CountTracks(0)
    UnsoloTrack(Selected_Track)
    Selected_Track = (Selected_Track + 1) % TrackNB
    Selected_Item = SelectFirstMediaItemOnSelected_Track()
    UpdateTransient_Selected_Item()
    Selected_Transient = 0
    SoloTrack(Selected_Track)
end

function SoloTrack(trackNumber)
    local track = reaper.GetTrack(0, trackNumber) 
    if track then
        reaper.SetMediaTrackInfo_Value(track, "I_SOLO", 1)  -- Solo the track
        reaper.UpdateArrange()  -- Update the arrange view to reflect changes
    end
end
function UnsoloTrack(trackNumber)
    local track = reaper.GetTrack(0, trackNumber) 
    if track then
        reaper.SetMediaTrackInfo_Value(track, "I_SOLO", 0)  -- Solo the track
        reaper.UpdateArrange()  -- Update the arrange view to reflect changes
    end
end

function CalculateAvrSpacing(onsetList)

    --check if there are onset 
    if onsetList == nil then
        return nil, "Need at least 3 onsets"
    end
    if #onsetList < 3  then
        return nil, "Need at least 3 onsets"
    end
    return ((onsetList[2]-onsetList[1]) + (onsetList[3]-onsetList[2]) + (onsetList[3]-onsetList[1])/2)*1000/3, ""
end

-- current transient/trackMarker management

-- delete take marker
function deleteAllTakeMarkers(item)
    -- Begin undo block
    -- reaper.Undo_BeginBlock()

    if item then
        -- Get the active take from the item
        local take = reaper.GetActiveTake(item)
        if take and not reaper.TakeIsMIDI(take) then
            -- Iterate over all take markers and remove them
            local numMarkers = reaper.GetNumTakeMarkers(take)
            for i = numMarkers-1, 0, -1 do
                reaper.DeleteTakeMarker(take, i)
            end
        end
    end

    -- End undo block
    -- reaper.Undo_EndBlock("Delete All Take Markers", -1)
end

-- To add manually
function AddOneTakeMarker_to_Selected_Item()
        reaper.Main_OnCommand(command["Item: Quick add take marker at play position or edit cursor"],0)
        UpdateTransient_Selected_Item()
end


-- Function to zoom into a position within a selected item
function ZoomIntoPosition(position)
    if position == nil then return end

    local zoom_window = 0.4

    -- Get the first selected item
    local item = reaper.GetSelectedMediaItem(0, 0)
    if item == nil then
        reaper.ShowMessageBox("No item selected", "Error", 0)
        return
    end

    -- Get the item start position
    local itemStart = reaper.GetMediaItemInfo_Value(item, "D_POSITION")
    -- Calculate the absolute position of the zoom point
    local zoomPosition = itemStart + position

    -- Calculate zoom start and end positions
    local zoomStart = zoomPosition - zoom_window --  before
    local zoomEnd = zoomPosition + zoom_window   --  after

    -- Set the arrange view to the zoom area
    reaper.BR_SetArrangeView(0, zoomStart, zoomEnd)

    -- Set the time selection range
    reaper.GetSet_LoopTimeRange(true, false, zoomStart, zoomEnd, false)

    -- Optionally set the edit cursor position to the start of the time selection
    reaper.SetEditCurPos(zoomStart + zoom_window, true, false)

    reaper.UpdateArrange()
end

function GetTakeMarkersPositions(item)
    -- Table to hold the positions of the take markers
    local markerPositions = {}

    -- Check if the item is valid
    if not item then
        reaper.ShowMessageBox("No item selected", "Error", 0)
        return
    end

    -- Get the first take of the given item
    local take = reaper.GetTake(item, 0) -- 0 gets the first take

    -- Check if the take is valid
    if not take then
        reaper.ShowMessageBox("Item has no takes", "Error", 0)
        return
    end

    -- Get the number of take markers
    TransientNB = reaper.GetNumTakeMarkers(take)

    -- Iterate through each take marker
    for i = 0, TransientNB - 1 do
        local pos = reaper.GetTakeMarker(take, i)
        -- Convert position from project time (seconds) to take time (seconds)
        -- and store it in the table, adjusting index to start from 1
        markerPositions[i + 1] = pos
    end

    return markerPositions -- Success
end

function UpdateTransient_Selected_Item()
    AddOnsetTo_OnsetList(GetTakeMarkersPositions(Selected_Item))
end

function Next_Transient()
    UpdateTransient_Selected_Item()
    Selected_Transient = ((Selected_Transient + 1) % (TransientNB+1))
    if Selected_Transient == 0 then Selected_Transient = 1 end
    ZoomIntoPosition(Onsets_Item[Selected_Track][Selected_Transient])
end

function Previous_Transient()
    UpdateTransient_Selected_Item()
    Selected_Transient = ((Selected_Transient - 1) % (TransientNB+1))
    if Selected_Transient == 0 then Selected_Transient = TransientNB end
    ZoomIntoPosition(Onsets_Item[Selected_Track][Selected_Transient])
end

-- ok its a strange behavior but I assume that onset are maintened by ref
--[[
function AddOnsetTo_OnsetList(onset)
    Onsets_Item[Selected_Track] = onset
end
]]

-- no ref, copying is better for safety
function AddOnsetTo_OnsetList(onset)
    Onsets_Item[Selected_Track] = {table.unpack(onset)}
end


-- use adjustTransientGuideSpacing instead : no follow the reaper implementation of dynamc split
function adjustTakeMarkerSpacing(mediaItem, minSliceLen)
    if not mediaItem then
        return false
    end

    local take = reaper.GetActiveTake(mediaItem)
    if not take then
        return false
    end

    -- Start undo block for undo-able actions
    -- reaper.Undo_BeginBlock()

    local numMarkers = reaper.GetNumTakeMarkers(take)
    if numMarkers < 2 then
        return false  -- Not enough markers to process spacing issues
    end

    -- First index 0 or 1 ???
    local pred_marker_idx = 0
    local pred_marker_pos = reaper.GetTakeMarker(take, pred_marker_idx)

    -- Loop through markers starting from the second one
    local i = 1
    while i < numMarkers do
        local markerPos = reaper.GetTakeMarker(take, i)
        if markerPos - pred_marker_pos < minSliceLen then
            -- Marker too close, remove it
            reaper.DeleteTakeMarkerByIndex(take, i)
            -- Do not increment i as the next marker now has the current index
            numMarkers = numMarkers - 1
        else
            -- Update the previous marker position and index
            pred_marker_idx = i
            pred_marker_pos = markerPos
            i = i + 1
        end
    end

    -- End undo block
    -- reaper.Undo_EndBlock("Adjust Take Marker Spacing", -1)

    return true
end

function adjustTransientGuideSpacing(mediaItem, minSliceLen)
    if not mediaItem then
        return false, "Invalid media item provided."
    end

    -- Assuming the transient guides and sample rate are globally accessible
    if #Selected_TransientsGuide < 2 then
        return false, "Not enough transient guides to process."
    end

    -- in ms , somme compensation
    local adjust = 55

    -- Convert minSliceLen from milliseconds to samples
    local minSampleLen = (minSliceLen / 1000 * Selected_SampleRate) - adjust

    local newTransientGuides = {}
    local lastPosition = 0  -- Initialize the last position to the start

    for i, samplesSinceLast in ipairs(Selected_TransientsGuide) do
        if i == 1 then
            -- Always include the first transient guide
            table.insert(newTransientGuides, samplesSinceLast)
            lastPosition = samplesSinceLast
        else
            if samplesSinceLast < minSampleLen then
                -- If the current guide is too close to the last, skip it
                -- You might also merge it by adding its value to the next guide instead of skipping
            else
                -- Accept this transient guide
                table.insert(newTransientGuides, samplesSinceLast)
                lastPosition = samplesSinceLast
            end
        end
    end
    -- Update the global transient guides with the new, adjusted list
    return setTransientMarkers(mediaItem,Selected_SampleRate,newTransientGuides), "Transient guides adjusted successfully."
end


-- From Transient API

function getTransientMarkers(item)
    if not item then return nil, "Item not provided" end

    local _, chunk = reaper.GetItemStateChunk(item, "", false)
    local sampleRate = chunk:match("TMINFO (%d+)")
    if sampleRate == nil then return nil,"No transient markers" end

    local TMs = chunk:gmatch("TM (%d[%d%s]*%d)")
    local rawTMs = {}
    
    for tms in TMs do
        for tm in tms:gmatch("(%d+)") do
            rawTMs[#rawTMs + 1] = tonumber(tm)
        end
    end

    return tonumber(sampleRate), rawTMs
end


function getTransientMarkersTimeStamps(item)
    local timestamps
    if item then
        local sampleRate, rawTMs = getTransientMarkers(item)
        -- reaper.ShowConsoleMsg("TMINFO "..tostring(sampleRate).."\n")
        -- for i,tm in ipairs(rawTMs) do  reaper.ShowConsoleMsg(" TM "..tostring(i)..": "..tostring(tm).."\n") end
        if sampleRate then
            timestamps = rawToTimestamps(sampleRate, rawTMs)
            -- for i,tm in ipairs(timestamps) do  reaper.ShowConsoleMsg(" TM "..tostring(i)..": "..tostring(tm)) end
            -- for i,tm in ipairs(timestampsToRaw(48000, timestamps)) do  reaper.ShowConsoleMsg(" TM "..tostring(i)..": "..tostring(tm)) end
        end
    end
    return timestamps
  end

function rawToTimestamps(sampleRate, rawTMs)
    local timestamps = {}
    local sum = 0

    for i, rawTM in ipairs(rawTMs) do
        sum = sum + rawTM
        local timestamp = sum / sampleRate
        -- Format to string keeping first 6 digits and convert back to number
        -- timestamp = tonumber(string.format("%.6f", timestamp))
        timestamps[i] = timestamp
    end

    return timestamps
end

function setTransientMarkers(item, sampleRate, rawTMs)
    if not item then return false, "Item not provided" end

    --clear transient guides to begin clean TODO manually clean
    reaper.Main_OnCommand(42027,0)

    -- very special behavior with inverted string to get the last '>'
    local _, chunk = reaper.GetItemStateChunk(item, "", false)
    local tmsString = "TMINFO " .. sampleRate .. "\n" .."TM " .. table.concat(rawTMs, " ") .. "\n"
    chunk = chunk:reverse()
    local newChunk,nb = chunk:gsub(">",">\n"..tmsString:reverse(),1)

    newChunk = newChunk:reverse()


    local res = reaper.SetItemStateChunk(item, newChunk, false)
    reaper.UpdateArrange()

    return res
end

function timestampsToRaw(sampleRate, timestamps)
    local rawTMs = {}
    local lastTimestamp = 0

    for i, timestamp in ipairs(timestamps) do
        local rawTM = math.floor(0.05 + (timestamp - lastTimestamp) * sampleRate) -- floor and +0.05 is to round value
        rawTMs[i] = rawTM
        lastTimestamp = timestamp
    end

    return rawTMs
end


-- Misc

function OpenHelp(url,section)
    return reaper.CF_ShellExecute(url.."#"..section)
end

function getTrackColorAndNegative(trackIndex)
    -- Get the track by index (0-based)
    local track = reaper.GetTrack(0, trackIndex)
    if track then
        -- Get the color of the track
        local color = reaper.GetMediaTrackInfo_Value(track, "I_CUSTOMCOLOR")
        if color ~= 0 then
            local r = color & 255
            local g = (color >> 8) & 255
            local b = (color >> 16) & 255

            -- Calculate the negative color
            local negative_r = 255 - r
            local negative_g = 255 - g
            local negative_b = 255 - b

            -- Pack the negative color back into an integer (0xBBGGRR)
            local negative_color = (negative_b << 16) | (negative_g << 8) | negative_r

            return negative_color, color
        else
            return 0, 0  -- No custom color set
        end
    end
    return nil  -- Track not found
end


-- Initialise the script
main()





-- GUI

-- Manage install and load UltrashallAPI to manage Chunk FX
function checkAPIAvailability()
    local status, err = pcall(dofile, lib_path .. "Core.lua")
    if not status then
        -- The API is not yet available; print the error and retry after a short delay
        -- reaper.ShowConsoleMsg("Ultraschall API is not available yet: " .. tostring(err) .. "\n")
        reaper.defer(checkAPIAvailability) -- Retry after a short delay
    else
        -- The API is available; you can now safely call its functions
        -- reaper.ShowConsoleMsg("Ultraschall API is loaded.\n")
        reaper.Main_OnCommand(reaper.NamedCommandLookup("_RS1c6ad1164e1d29bb4b1f2c1acf82f5853ce77875"),0)
    end
end
function loadInstallGuiAPI()
    lib_path = reaper.GetExtState("Lokasenna_GUI", "lib_path_v2")
    if not lib_path or lib_path == "" then
        -- reaper.MB("Couldn't load the Lokasenna_GUI library. Please install 'Lokasenna's GUI library v2 for Lua', available on ReaPack, then run the 'Set Lokasenna_GUI v2 library path.lua' script in your Action List.", "Whoops!", 0)
        -- return
    end
  local status, err = pcall(function()dofile(lib_path .. "Core.lua")end)
  if not status then
      reaper.MB("Couldn't load the Lokasenna_GUI library. Please install 'Lokasenna's GUI library v2 for Lua', available on ReaPack, then run the 'Set Lokasenna_GUI v2 library path.lua' script in your Action List.", "Whoops!", 0)
      print("An error occurred: " .. err)
    --   local ok,error = reaper.ReaPack_AddSetRepository("Get Lokasenna_GUI library-API"
    --                                                 ,"https://raw.githubusercontent.com/ReaTeam/ReaScripts/master/index.xml"
    --                                                 , true
    --                                                 , 2)
    --   if ok then
    --     reaper.ReaPack_ProcessQueue(true)
    --     -- Waiting to API succesfully download
    --     checkAPIAvailability()
    --   end
    reaper.Main_OnCommand(reaper.NamedCommandLookup("_RS1c6ad1164e1d29bb4b1f2c1acf82f5853ce77875"),0)
  end
end

-- import library for GUI
loadInstallGuiAPI()


-- loadfile(lib_path .. "Core.lua")()

-- load class

GUI.req("Classes/Class - Textbox.lua")()
GUI.req("Classes/Class - Frame.lua")()
GUI.req("Classes/Class - Button.lua")()
GUI.req("Classes/Class - Tabs.lua")()
GUI.req("Classes/Class - Label.lua")()
GUI.req("Classes/Class - Options.lua")()
GUI.req("Classes/Class - Slider.lua")()
-- If any of the requested libraries weren't found, abort the script.
if missing_lib then return 0 end

-- Window Option

GUI.name = "Analyse Dsync - Onset extraction Tool"
GUI.x, GUI.y, GUI.w, GUI.h = 0, 0, 600, 700
GUI.anchor, GUI.corner = "screen", "BL"



-- GUI variable and constant

-- header is on the layer 01 02 03
-- baground frame on the layer 100
-- tab 1 on the layer 11 12 12
-- tab 2 on the layer 21 22 23
-- tab 3 on the layer 31 32 33




-- Main content

-- Header

GUI.New("header_frame", "Frame", {
    z = 2,
    x = 16,
    y = 16,
    w = 448,
    h = 100,
    shadow = true,
    fill = false,
    color = "elm_frame",
    bg = "wnd_bg",
    round = 0,
    text = "",
    txt_indent = 0,
    txt_pad = 0,
    pad = 4,
    font = 4,
    col_txt = "txt"
})

GUI.New("previous_item", "Button", {
    z = 1,
    x = 50,
    y = 80,
    w = 48,
    h = 24,
    caption = "<-",
    font = 3,
    col_txt = "txt",
    col_fill = "elm_frame",
    func = function() Previous_item() GUIupdateHeader() end
})

GUI.New("next_item", "Button", {
    z = 1,
    x = 370,
    y = 80,
    w = 48,
    h = 24,
    caption = "->",
    font = 3,
    col_txt = "txt",
    col_fill = "elm_frame",
    func = function() Next_item() GUIupdateHeader() end
})


GUI.New("item_name", "Textbox", {
    z = 1,
    x = 192,
    y = 32,
    w = 250,
    h = 20,
    caption = "item name : ",
    cap_pos = "left",
    font_a = 4,
    font_b = "monospace",
    color = "txt",
    bg = "wnd_bg",
    shadow = true,
    pad = 4,
    undo_limit = 0,

    retval = reaper.GetTakeName(reaper.GetActiveTake(Selected_Item))
})

GUI.New("track_number", "Textbox", {
    z = 1,
    x = 80,
    y = 32,
    w = 40,
    h = 20,
    caption = "track : ",
    cap_pos = "left",
    font_a = 4,
    font_b = "monospace",
    color = "txt",
    bg = "wnd_bg",
    shadow = true,
    pad = 4,
    undo_limit = 0,

    retval = (Selected_Track+1).."/"..TrackNB
})


GUI.New("spacing_value", "Textbox", {
    z = 1,
    x = 250,
    y = 80,
    w = 96,
    h = 20,
    caption = "estimated spacing (ms) : ",
    cap_pos = "left",
    font_a = 3,
    font_b = "monospace",
    color = "txt",
    bg = "wnd_bg",
    shadow = true,
    pad = 4,
    undo_limit = 20
})

-- Nav
GUI.New("tool_select", "Tabs", {
    z = 1,
    x = 16,
    y = 128,
    w = 448,
    caption = "tool_select",
    optarray = {"Spectral Cleaning", "Transient Pick", "Export"},
    tab_w = 136,
    tab_h = 30,
    pad = 3,
    font_a = 3,
    font_b = 4,
    col_txt = "txt",
    col_tab_a = "wnd_bg",
    col_tab_b = "tab_bg",
    bg = "elm_bg",
    fullwidth = false
})

-- content

GUI.New("content_frame", "Frame", {
    z = 100,
    x = 16,
    y = 160,
    w = 448,
    h = 400,
    shadow = true,
    fill = false,
    color = "elm_frame",
    bg = "wnd_bg",
    round = 0,
    text = "",
    txt_indent = 0,
    txt_pad = 0,
    pad = 4,
    font = 4,
    col_txt = "txt"
})

-- Content Spectral cleaning

-- Core Spectral cleaning

GUI.New("done", "Button", {
    z = 11,
    x = 288,
    y = 272,
    w = 90,
    h = 50,
    caption = "done",
    font = 3,
    col_txt = "txt",
    col_fill = "elm_frame",

    func = function () FreezeTrack(Selected_Track) end
})

GUI.New("sound_print", "Button", {
    z = 11,
    x = 288,
    y = 208,
    w = 90,
    h = 50,
    caption = "get sound print",
    font = 3,
    col_txt = "txt",
    col_fill = "elm_frame",
    func = GetSoundPrint
})

--[[
-- Not sure of utility
GUI.New("freeze_track", "Checklist", {
    z = 11,
    x = 262,
    y = 336,
    w = 145,
    h = 35,
    caption = "",
    optarray = {"Freeze Track :"},
    dir = "v",
    pad = 4,
    font_a = 2,
    font_b = 3,
    col_txt = "txt",
    col_fill = "elm_fill",
    bg = "wnd_bg",
    frame = true,
    shadow = true,
    swap = true,
    opt_size = 20
})
]]

GUI.New("core_sc_bg_frame", "Frame", {
    z = 12,
    x = 262,
    y = 192,
    w = 145,
    h = 150,
    shadow = false,
    fill = false,
    color = "elm_frame",
    bg = "wnd_bg",
    round = 0,
    text = "",
    txt_indent = 0,
    txt_pad = 0,
    pad = 4,
    font = 4,
    col_txt = "txt"
})

GUI.New("sc_help", "Button", {
    z = 11,
    x = 395,
    y = 192,
    w = 12,
    h = 12,
    caption = "?",
    font = 4,
    col_txt = "txt",
    col_fill = "elm_outline",

    func = function() OpenHelp(DOC_URL,"Spectral Cleaning") end
})

-- options Spectral cleaning
-- Option lable_frame
GUI.New("opt_sc_label_frame", "Frame", {
    z = 11,
    x = 48,
    y = 192,
    w = 116,
    h = 32,
    shadow = false,
    fill = false,
    color = "elm_frame",
    bg = "wnd_bg",
    round = 0,
    text = "",
    txt_indent = 0,
    txt_pad = 0,
    pad = 4,
    font = 4,
    col_txt = "txt"
})

GUI.New("options_sc", "Label", {
    z = 11,
    x = 52,
    y = 197,
    caption = "Setting",
    font = 2,
    color = "txt",
    bg = "wnd_bg",
    shadow = false
})

GUI.New("opt_sc_bg_frame", "Frame", {
    z = 12,
    x = 48,
    y = 192,
    w = 116,
    h = 114,
    shadow = false,
    fill = false,
    color = "elm_frame",
    bg = "wnd_bg",
    round = 0,
    text = "",
    txt_indent = 0,
    txt_pad = 0,
    pad = 4,
    font = 4,
    col_txt = "txt"
})

GUI.New("peaks_settings", "Button", {
    z = 11,
    x = 64,
    y = 240,
    w = 85,
    h = 50,
    caption = "peaks settings",
    font = 3,
    col_txt = "txt",
    col_fill = "elm_frame",
    func = showPeakDisplaySetting
})

-- Content Transient pick

-- Core Transient pick

GUI.New("core_tp_bg_frame", "Frame", {
    z = 22,
    x = 262,
    y = 192,
    w = 145,
    h = 320,
    shadow = false,
    fill = false,
    color = "elm_frame",
    bg = "wnd_bg",
    round = 0,
    text = "",
    txt_indent = 0,
    txt_pad = 0,
    pad = 4,
    font = 4,
    col_txt = "txt"
})



GUI.New("tp_help", "Button", {
    z = 21,
    x = 395,
    y = 192,
    w = 12,
    h = 12,
    caption = "?",
    font = 4,
    col_txt = "txt",
    col_fill = "elm_outline",

    func = function() OpenHelp(DOC_URL,"Transient Picking") end
})


GUI.New("min_slice", "Slider", {
    z = 21,
    x = 272,
    y = 224,
    w = 117,
    caption = "Min slice length (ms) :",
    min = 20,
    max = 1000,
    defaults = {500},
    inc = 1,
    dir = "h",
    font_a = 3,
    font_b = 4,
    col_txt = "txt",
    col_fill = "elm_fill",
    bg = "wnd_bg",
    show_handles = true,
    show_values = true,
    cap_x = 0,
    cap_y = 0
})
--[[ 
GUI.New("min_silence", "Slider", {
    z = 21,
    x = 272,
    y = 288,
    w = 117,
    caption = "Min silence length :",
    min = 20,
    max = 5000,
    defaults = {500},
    inc = 1,
    dir = "h",
    font_a = 3,
    font_b = 4,
    col_txt = "txt",
    col_fill = "elm_fill",
    bg = "wnd_bg",
    show_handles = true,
    show_values = true,
    cap_x = 0,
    cap_y = 0
})
]]

GUI.New("transient_sensivity", "Slider", {
    z = 21,
    x = 272,
    y = 284,
    w = 117,
    caption = "Sensivity (%) :",
    min = 0.0,
    max = 100.0,
    -- divide by 10 on float ??? default is multiply by incr and add min
    defaults = {500.0},
    inc = 0.1,
    dir = "h",
    font_a = 3,
    font_b = 4,
    col_txt = "txt",
    col_fill = "elm_fill",
    bg = "wnd_bg",
    show_handles = true,
    show_values = true,
    cap_x = 0,
    cap_y = 0
})

GUI.New("transient_treshold", "Slider", {
    z = 21,
    x = 272,
    y = 348,
    w = 117,
    caption = "Treshold (dB) :",
    min = -60.0,
    max = 0.0,
    -- default : -17, (-60+43*10) ??? default is multiply by incr and add min
    defaults = {430.0},
    inc = 0.1,
    dir = "h",
    font_a = 3,
    font_b = 4,
    col_txt = "txt",
    col_fill = "elm_fill",
    bg = "wnd_bg",
    show_handles = true,
    show_values = true,
    cap_x = 0,
    cap_y = 0
})

GUI.New("update_transients", "Checklist", {
    z = 21,
    x = 262,
    y = 396,
    w = 145,
    h = 35,
    caption = "",
    optarray = {"Update transients :"},
    dir = "v",
    pad = 4,
    font_a = 2,
    font_b = 3,
    col_txt = "txt",
    col_fill = "elm_fill",
    bg = "wnd_bg",
    frame = true,
    shadow = true,
    swap = true,
    opt_size = 20
})


GUI.New("previous_transient", "Button", {
    z = 21,
    x = 272,
    y = 460,
    w = 24,
    h = 24,
    caption = "◄",
    font = 3,
    col_txt = "txt",
    col_fill = "elm_frame",
    func = function() Previous_Transient() GUIupdateTransient() end
})

GUI.New("next_transient", "Button", {
    z = 21,
    x = 368,
    y = 460,
    w = 24,
    h = 24,
    caption = "►",
    font = 3,
    col_txt = "txt",
    col_fill = "elm_frame",
    func = function() Next_Transient() GUIupdateTransient() end,
})

GUI.New("transient_number", "Textbox", {
    z = 21,
    x = 310,
    y = 460,
    w = 45,
    h = 20,
    caption = "transient :",
    cap_pos = "top",
    font_a = 3,
    font_b = "monospace",
    color = "txt",
    bg = "wnd_bg",
    shadow = true,
    pad = 4,
    undo_limit = 0,

    retval = (Selected_Transient).."/"..TransientNB
})

GUI.New("add_transient", "Button", {
    z = 21,
    x = 324,
    y = 487,
    w = 15,
    h = 15,
    caption = "♦",
    font = 3,
    col_txt = "txt",
    col_fill = "elm_frame",
    func = function() AddOneTakeMarker_to_Selected_Item() GUIupdateTransient() end
})

-- options Transient pick

GUI.New("opt_tp_label_frame", "Frame", {
    z = 22,
    x = 48,
    y = 192,
    w = 116,
    h = 32,
    shadow = false,
    fill = false,
    color = "elm_frame",
    bg = "wnd_bg",
    round = 0,
    text = "",
    txt_indent = 0,
    txt_pad = 0,
    pad = 4,
    font = 4,
    col_txt = "txt"
})

GUI.New("opt_tp_bg_frame", "Frame", {
    z = 23,
    x = 48,
    y = 192,
    w = 116,
    h = 114,
    shadow = false,
    fill = false,
    color = "elm_frame",
    bg = "wnd_bg",
    round = 0,
    text = "",
    txt_indent = 0,
    txt_pad = 0,
    pad = 4,
    font = 4,
    col_txt = "txt"
})

GUI.New("options_tp", "Label", {
    z = 21,
    x = 80,
    y = 197,
    caption = "Settings",
    font = 2,
    color = "txt",
    bg = "wnd_bg",
    shadow = false
})

GUI.New("dynamic_split_settings", "Button", {
    z = 21,
    x = 60,
    y = 240,
    w = 95,
    h = 24,
    caption = "dynamic split",
    font = 3,
    col_txt = "txt",
    col_fill = "elm_frame",
    func = showDynamicSplitItems
})

GUI.New("transient_detect_settings", "Button", {
    z = 21,
    x = 60,
    y = 272,
    w = 95,
    h = 24,
    caption = "transient detect",
    font = 3,
    col_txt = "txt",
    col_fill = "elm_frame",
    func = showTransientDetectionSetting
})

-- Content Export

-- Core Export

GUI.New("core_e_bg_frame", "Frame", {
    z = 32,
    x = 262,
    y = 192,
    w = 145,
    h = 140,
    shadow = false,
    fill = false,
    color = "elm_frame",
    bg = "wnd_bg",
    round = 0,
    text = "",
    txt_indent = 0,
    txt_pad = 0,
    pad = 4,
    font = 4,
    col_txt = "txt"
})


GUI.New("e_help", "Button", {
    z = 31,
    x = 395,
    y = 192,
    w = 12,
    h = 12,
    caption = "?",
    font = 4,
    col_txt = "txt",
    col_fill = "elm_outline",

    func = function() OpenHelp(DOC_URL,"Export") end
})

GUI.New("export_selected", "Button", {
    z = 31,
    x = 276.0,
    y = 212.0,
    w = 115,
    h = 24,
    caption = "export selected item",
    font = 3,
    col_txt = "txt",
    col_fill = "elm_frame",

    func = function() UpdateTransient_Selected_Item() exportTakeMarkersToCSV(Onsets_Item[Selected_Track])end
})

GUI.New("export_all", "Button", {
    z = 31,
    x = 275.0,
    y = 267.0,
    w = 115,
    h = 24,
    caption = "export all items",
    font = 3,
    col_txt = "txt",
    col_fill = "elm_frame",

    func =  function() 
                for i = 0, TrackNB - 1 do
                    UpdateTransient_Selected_Item()
                    Next_item()
                end
                exportAll()
            end
})

-- options Export

GUI.New("opt_e_label_frame", "Frame", {
    z = 32,
    x = 48,
    y = 192,
    w = 116,
    h = 32,
    shadow = false,
    fill = false,
    color = "elm_frame",
    bg = "wnd_bg",
    round = 0,
    text = "",
    txt_indent = 0,
    txt_pad = 0,
    pad = 4,
    font = 4,
    col_txt = "txt"
})

GUI.New("opt_e_bg_frame", "Frame", {
    z = 33,
    x = 48,
    y = 192,
    w = 116,
    h = 114,
    shadow = false,
    fill = false,
    color = "elm_frame",
    bg = "wnd_bg",
    round = 0,
    text = "",
    txt_indent = 0,
    txt_pad = 0,
    pad = 4,
    font = 4,
    col_txt = "txt"
})

GUI.New("options_e", "Label", {
    z = 31,
    x = 80,
    y = 197,
    caption = "Setting",
    font = 2,
    color = "txt",
    bg = "wnd_bg",
    shadow = false
})

-- TODO not yet
--[[
GUI.New("format", "Textbox", {
    z = 31,
    x = 57.0,
    y = 247.0,
    w = 96,
    h = 20,
    caption = "format :",
    cap_pos = "top",
    font_a = 3,
    font_b = "monospace",
    color = "txt",
    bg = "wnd_bg",
    shadow = true,
    pad = 4,
    undo_limit = 20
})
]]

-- function for the GUI

-- Telling the tabs which z layers to display
-- See Classes/Tabs.lua for more detail
GUI.elms.tool_select:update_sets(
  --  Tab
  --               Layers
  {     
    [1] =     {11,12,13},
    [2] =     {21,22,23},
    [3] =     {31,32,33},
  }
)


function GUIupdateHeader()
    GUIupdateTransient()
    GUI.Val("track_number",(Selected_Track+1).."/"..TrackNB)
    GUI.Val("item_name",reaper.GetTakeName(reaper.GetActiveTake(Selected_Item)))
end

function GUIupdateTransient()
    GUI.Val("transient_number",(Selected_Transient).."/"..TransientNB)
    local spacing,txt = CalculateAvrSpacing(Onsets_Item[Selected_Track])
    if spacing == nil then
        GUI.Val("spacing_value",txt)
    else
        GUI.Val("spacing_value",spacing)
    end
end

function GUIupdateTransients()
    -- Print_console(tostring(GUI.Val("update_transients")))
    if GUI.Val("update_transients") then
        --[[
        if getTransientMarkers(Selected_Item) then
            deleteAllTakeMarkers(Selected_Item)
            ConvertTransientGuidesToTakeMarkers()
            UpdateTransient_Selected_Item()
            reaper.Main_OnCommand(command["Clear transient guides"],0)
            GUIupdateTransient()
        end
        ]]
        Selected_SampleRate,Selected_TransientsGuide = getTransientMarkers(Selected_Item)
        if Selected_SampleRate then
            deleteAllTakeMarkers(Selected_Item)
            ConvertTransientGuidesToTakeMarkers()
            UpdateTransient_Selected_Item()
            reaper.Main_OnCommand(command["Clear transient guides"],0)
            GUIupdateTransient()
        end
    end
end

-- sliders

Previous_slider = GUI.Val("min_slice")
Previous_sliderA = GUI.Val("transient_sensivity")
Previous_sliderB = GUI.Val("transient_treshold")

function GUIupdateSliders()
    local slider = GUI.Val("min_slice")
    local sliderA = GUI.Val("transient_sensivity")
    local sliderB = GUI.Val("transient_treshold")

    -- Print_console(tostring(sliderA).." "..tostring(sliderB).."\n")

    if (sliderA ~= Previous_sliderA) or (sliderB ~= Previous_sliderB) or (slider ~= Previous_slider) then
        Previous_sliderA = sliderA
        Previous_sliderB = sliderB
        Previous_slider = slider
        -- reaper.Main_OnCommand(command["Clear transient guides"],0)
        updateTransientCalculation(sliderA/100,sliderB)
        updateDynamicSplitPreset(slider,20)
        updateDynamicSplit(slider,20)
        CalculateTransientGuides()
        adjustTransientGuideSpacing(Selected_Item,slider)
    end
    
end

-- called everytime
GUI.func = function()
    GUIupdateTransients()
    GUIupdateSliders()
end
GUI.freq = 0.7


-- launch GUI
GUI.Init()
GUI.Main()


