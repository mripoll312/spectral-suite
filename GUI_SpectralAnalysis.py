"""Author: Felix Winkler (TU Berlin)
1.02 (19.11.2021)
n: metadatafile check: detects mismatching Wavelengths in metadatafile and reference_spectra
1.01 (04.11.2021)
f: metadata can be utf-8 or ascii
n: datatoolbox failed message leads to 'check metadata file'
0.99 (06.05.2021) start to add mac usage possible
n: another file explorer oopener for mac
n: switch to / for mac
n: FileDialog takes for mac the non-nativedialog (maybe only bug with macos catalina)
0.981 (22.03.2021)
f: metadata-check now detects ' because the datatoolbox cant handle this
0.98 (04.03.2021)
c: metadata-check now also looks for encoding
0.97 (18.02.2021)
n: check metadata file
0.96 (14.02.2021)
f: checkpath doesnt (dis-)enabled the correct buttons
0.95 (20.01.2021)
c: path for Form is now relative to Start-Bat location
0.94 (16.11.2020)
c: metafile -> metadata file [only in Fomular]
c: gen5split: create processedfolder if doesnt exist
0.93 (11.09.2020)
c: select gen folder/file -> select files with multiselect
n: option to change the destination folder for processed_data
n: option to select/change the metafile
n: clear log and clear all(new) option
0.92
c: call gen5split over subprocess.popen
n: live log for gensplit and data_toolbox
c: lock all buttons during gensplit and data_toolbox
0.91
n: btn openmetafile
c: filepath textedit -> lineedit (.toplaintext() -> .text() )
0.9
"""
# Fix: On a fresh install, open anaconda powershell and enter:
#python -m pip install --upgrade pyqt5
#pip install --upgrade --force-reinstall numpy


import subprocess, sys
from subprocess import Popen
import os
from shutil import copyfile
from PyQt5.QtWidgets import *
from PyQt5 import uic
from gen5_reader.split import split
import pandas as pd
import time
import chardet

# Collect Path of the GUI_graphics and open it
path = os.path.realpath(__file__) 
qtCreatorFile = path.replace("GUI_SpectralAnalysis.py","Form_SpectralAnalysis.ui")# Enter file here.

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

# fills the big right textbox with feedback for the user from the GUI, gen5split and datatoolbox
def writeLog(self,input):
	self.ui.lbl_te_log.append(input)
	#Refresh displayed log in GUI
	QApplication.processEvents()
	
# verify the selected values(paths) and dis-/enable controls
def checkpath(self):
	self.ui.btn_new.setEnabled(True)
	self.ui.btn_opendirectory.setEnabled(True)
	self.ui.btn_selectrawfiles.setEnabled(True)
	self.ui.btn_clearrawfiles.setEnabled(True)
	self.ui.btn_selectdestination.setEnabled(True)
	self.ui.btn_checkfiles.setEnabled(True)
	self.ui.btn_clearlog.setEnabled(True)	
	rawfiles = self.ui.lbl_te_rawfiles.toPlainText()
	processedfolder = self.ui.lbl_le_destinationfolder.text()
	metafile = self.ui.lbl_le_metafile.text()
	if rawfiles != "":
		#Check if platereader-files are selected
		if rawfiles.find("Spectral_Analysis/raw_data") != -1:
			self.ui.btn_gensplit.setEnabled(True)
		else:
			self.ui.btn_gensplit.setEnabled(False)
			writeLog(self,"Platereader Gen5.txt-files have to be located in: Spectral_Analysis/raw_data/...")
		#Check if processed_data exist for platereader files
		for genfile in rawfiles.splitlines():
			if os.path.isfile(processedfolder+"/"+genfile.split("/")[-1].split(".txt")[0]+"_Spectrum.tsv") == True:
				self.ui.cb_gensplit.setChecked(True)
			else:
				self.ui.cb_gensplit.setChecked(False)
				break	
	else:
		self.ui.btn_gensplit.setEnabled(False)
		self.ui.cb_gensplit.setChecked(False)
		self.ui.btn_selectmetafile.setEnabled(False)
		self.ui.cb_metadata.setChecked(False)
		self.ui.btn_datatoolbox.setEnabled(False)
		self.ui.btn_openmeta.setEnabled(False)
		self.ui.btn_checkmetadata.setEnabled(False)
		self.ui.cb_datatoolbox.setChecked(False) 
	# Check if destination-folder is selected
	if processedfolder != "" :
		if processedfolder.find("Spectral_Analysis/processed_data") != -1:
			self.ui.btn_selectmetafile.setEnabled(True)
			if 	self.ui.cb_gensplit.isChecked() == False:
				if os.path.isdir(processedfolder) == True:
					files = [f for f in os.listdir(processedfolder) if os.path.isfile(os.path.join(processedfolder, f))]
					if files:
						self.ui.cb_gensplit.setChecked(True)
		else:
			self.ui.btn_selectmetafile.setEnabled(False)
			writeLog(self,"Destination has to be located in: Spectral_Analysis/processed_data/...")
	else:
		self.ui.btn_selectmetafile.setEnabled(False)		
	#Check if metafile exist
	if metafile != "":
		self.ui.cb_metadata.setChecked(True)
		self.ui.btn_openmeta.setEnabled(True)
		self.ui.btn_checkmetadata.setEnabled(True)
	else:
		self.ui.cb_metadata.setChecked(False)
		self.ui.btn_openmeta.setEnabled(False)
		self.ui.btn_checkmetadata.setEnabled(False)
	#Check if gensplit, metafile and destination-folder exist
	if self.ui.cb_metadata.isChecked() and self.ui.cb_gensplit.isChecked() and processedfolder!="":
		self.ui.btn_datatoolbox.setEnabled(True)
	else:
		self.ui.btn_datatoolbox.setEnabled(False)		
	#Check if data_toolbox files exist
	if os.path.isdir(processedfolder+"/Graphs") == True:
		self.ui.cb_datatoolbox.setChecked(True)
	else:
		self.ui.cb_datatoolbox.setChecked(False)
	writeLog(self,"Files checked.")

# calls the gen5split function; works only if gen5split is installed		
def processGenSplit(self,rawfile,processedpath):
	process = subprocess.Popen(["gen5split", "-sn",rawfile], stdout=subprocess.PIPE,stderr=subprocess.PIPE, cwd=processedpath)
	#Create Log (realtime, line by line)
	while True:
		output = process.stderr.readline()
		if output == b'' and process.poll() is not None:
			break
		if output:
			output=str(output.strip())
			print(output)
			writeLog(self,output.replace("b'","").replace("'","").replace(r"\r\n",""))
	writeLog(self,"")
	rc = process.poll()
	print(rc)
	#Refresh displayed log in GUI
	QApplication.processEvents()	
	
# disable all buttons, to prevent click-events while code is running
def disableButtons(self):
	self.ui.btn_new.setEnabled(False)
	self.ui.btn_opendirectory.setEnabled(False)
	self.ui.btn_selectrawfiles.setEnabled(False)
	self.ui.btn_clearrawfiles.setEnabled(False)
	self.ui.btn_selectdestination.setEnabled(False)
	self.ui.btn_gensplit.setEnabled(False)
	self.ui.btn_selectmetafile.setEnabled(False)
	self.ui.btn_openmeta.setEnabled(False)
	self.ui.btn_checkmetadata.setEnabled(False)
	self.ui.btn_datatoolbox.setEnabled(False)
	self.ui.btn_checkfiles.setEnabled(False)
	self.ui.btn_clearlog.setEnabled(False)
	
# checks the metadata file for error
def checkMetadata(self,metafilepath):
	start_time = time.time()
	with open(metafilepath, 'rb') as rawdata:
		result = chardet.detect(rawdata.read(100000))
	if result['encoding'] != 'utf-8' and result['encoding'] != 'ascii':
		writeLog(self,"Metadata file is not encoded with utf-8. Please make sure that no Umlauts were used.")
		writeLog(self,str(result['encoding']))
		return
	elif result['confidence'] < 0.9:
		print("WARNING: Confidence for your metadata codec is low. Please make sure that no Umlauts were used.")
	df = pd.read_csv(metafilepath, skip_blank_lines=True).dropna(how="all")
	df.index += 2
	#progress Bar
	maxcount = round(len(df.index)*1.1)
	count = 0
	self.ui.progressBar.setMaximum(maxcount)
	self.ui.progressBar.setValue(count)
	errorcount = 0
	path = os.path.realpath(__file__)
	Wells = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","B1","B2","B3","B4","B5","B6","B7","B8","B9","B10","B11","B12","C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","D1","D2","D3","D4","D5","D6","D7","D8","D9","D10","D11","D12","E1","E2","E3","E4","E5","E6","E7","E8","E9","E10","E11","E12","F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","G1","G2","G3","G4","G5","G6","G7","G8","G9","G10","G11","G12","H1","H2","H3","H4","H5","H6","H7","H8","H9","H10","H11","H12"]
	ref_spectra_row = 0
	ref_spectra_lst = []
	writeLog(self,"\nmetadata-file check start...")	
	for row in df.index:
		csv_row = row # row starts with 0 and the first row are the header data
		if pd.isna(df.Condition_Name[row]) == False:
			# check for ' in Condition_Name
			strapo = str(df.Condition_Name[row])
			if strapo.find("'") != -1:
				writeLog(self,"Row: " + str(csv_row) + " [Condition_Name] does not contain '-character (apostrophe).")
				errorcount += 1				
			# check the general Columns for missing data
			if pd.isna(df.Unique_Sample_ID[row]):
				writeLog(self,"Row: " + str(csv_row) + " [Unique_Sample_ID] is missing.")
				errorcount += 1
			else:
				# check for ' in Unique_Sample_ID			
				strapo = str(df.Unique_Sample_ID[row])
				if strapo.find("'") != -1:
					writeLog(self,"Row: " + str(csv_row) + " [Unique_Sample_ID] does not contain '-character (apostrophe).")
					errorcount += 1					
			tsv_found = False
			if pd.isna(df.Data_File[row]):
				writeLog(self,"Row: " + str(csv_row) + " [Data_File] is missing.")
				errorcount += 1
			else:
				# check for ' in Data_File			
				strapo = str(df.Data_File[row])
				if strapo.find("'") != -1:
					writeLog(self,"Row: " + str(csv_row) + " [Data_File] does not contain '-character (apostrophe).")
					errorcount += 1				
				# data file has to be in processed_data
				tsv_path = self.ui.lbl_le_destinationfolder.text() + "/" + str(df.Data_File[row])
				if os.path.isfile(tsv_path)==False:
					writeLog(self,"Row: " + str(csv_row) + " [Data_File]: "+str(df.Data_File[row])+" can not be found in your 'processed_data'-folder.")
					errorcount += 1
				else:
					tsv_found = True
			if pd.isna(df.Well_Number[row]):
				writeLog(self,"Row: " + str(csv_row) + " [Well_Number] is missing.")
				errorcount += 1
			else:
				if (df.Well_Number[row] in Wells) == False:
					writeLog(self,"Row: " + str(csv_row) + " [Well_Number]: "+str(df.Well_Number[row])+" has to be between A1 and H12.")
					errorcount += 1
				elif tsv_found:
					# check if the tsv file has data in this Well
					df_tsv = pd.read_csv(tsv_path,sep='\t',usecols = [df.Well_Number[row]])
					for data_value in df_tsv[df.Well_Number[row]].unique():
						if pd.isna(data_value):
							writeLog(self,"Row: " + str(csv_row) + " [Well_Number]: "+str(df.Well_Number[row])+" has no data in "+str(df.Data_File[row]))
							errorcount += 1
							break
			if pd.isna(df.Replicate[row]):
				writeLog(self,"Row: " + str(csv_row) + " [Replicate] is missing.")
				errorcount += 1
			if pd.isna(df.Wavelength_Start[row]):
				writeLog(self,"Row: " + str(csv_row) + " [Wavelength_Start] is missing.")
				errorcount += 1
			if pd.isna(df.Wavelength_End[row]):
				writeLog(self,"Row: " + str(csv_row) + " [Wavelength_End] is missing.")
				errorcount += 1				
			# check the column for the not-blank-data
			if not(pd.isna(df.Time_Point[row]) or pd.isna(df.Blank_Unique_Sample_ID[row]) or pd.isna(df.Isosbestic_Point[row]) or pd.isna(df.Reference_Spectrum_Substrate[row]) or pd.isna(df.Reference_Spectrum_Product[row]) or pd.isna(df.Total_Concentration[row])):
				if pd.isna(df.Time_Point[row]):
					writeLog(self,"Row: " + str(csv_row) + " [Time_Point] is missing.")
					errorcount += 1
				if pd.isna(df.Blank_Unique_Sample_ID[row]):
					writeLog(self,"Row: " + str(csv_row) + " [Blank_Unique_Sample_ID] is missing.")
					errorcount += 1
				else:
					# Blank has to be defined in this file
					if (df.Blank_Unique_Sample_ID[row] in df.Unique_Sample_ID.unique()) == False:
						writeLog(self,"Row: " + str(csv_row) + " [Blank_Unique_Sample_ID]: "+str(df.Blank_Unique_Sample_ID[row])+" can not be found in [Unique_Sample_ID].")
						errorcount += 1
				if pd.isna(df.Isosbestic_Point[row]):
					writeLog(self,"Row: " + str(csv_row) + " [Isosbestic_Point] is missing.")
					errorcount += 1
				if pd.isna(df.Reference_Spectrum_Substrate[row]):
					writeLog(self,"Row: " + str(csv_row) + " [Reference_Spectrum_Substrate] is missing.")
					errorcount += 1
				else:
					# find spectra.csv
					spectra_path = str(df.Reference_Spectrum_Substrate[row])
					if spectra_path.find("../../reference_spectra/")==0:
						if sys.platform == "win32":
							mpath = path.replace("packages/GUI/GUI_SpectralAnalysis.py","reference_spectra/") + spectra_path.replace("../../reference_spectra/","")
						else:
							mpath = path.replace("packages/GUI/GUI_SpectralAnalysis.py","reference_spectra/") + spectra_path.replace("../../reference_spectra/","")						
						if os.path.isfile(mpath) == False:
							writeLog(self,"Row: " + str(csv_row) + " [Reference_Spectrum_Substrate]: "+spectra_path.replace("../../reference_spectra/","")+" cant be found in 'reference_spectra'-folder.")
							errorcount += 1
						else:
							rs_exist = False
							for rs_row in range(len(ref_spectra_lst)):
								if ref_spectra_lst[rs_row][0] == mpath:
									rs_exist = True
									break
							if rs_exist == False:					
								ref_spectra = pd.read_csv(mpath, skip_blank_lines=True).dropna(how="all")
								ref_spectra.index += 1
								ref_spectra_lst.append([mpath,ref_spectra.Wavelength[1],ref_spectra.Wavelength[len(ref_spectra.index)]])
								rs_row = len(ref_spectra_lst)-1
							if df.Wavelength_Start[row] < ref_spectra_lst[rs_row][1]:
								writeLog(self,"Row: " + str(csv_row) + " [Reference_Spectrum_Substrate]: "+spectra_path.replace("../../reference_spectra/","")+" Wavelength starts at " +str(ref_spectra_lst[rs_row][1])+" instead of "+ str(df.Wavelength_Start[row]))
								errorcount += 1											
							if df.Wavelength_End[row] > ref_spectra_lst[rs_row][2]:
								writeLog(self,"Row: " + str(csv_row) + " [Reference_Spectrum_Substrate]: "+spectra_path.replace("../../reference_spectra/","")+" Wavelength ends at " +str(ref_spectra_lst[rs_row][2])+" instead of " + str(df.Wavelength_End[row]))
								errorcount += 1										
					else:
						writeLog(self,"Row: " + str(csv_row) + " [Reference_Spectrum_Substrate] has to start with: ../../reference_spectra/")
						errorcount += 1
				if pd.isna(df.Reference_Spectrum_Product[row]):
					writeLog(self,"Row: " + str(csv_row) + " [Reference_Spectrum_Product] is missing.")
					errorcount += 1
				else:
					# find spectra.csv
					spectra_path = str(df.Reference_Spectrum_Product[row])
					if spectra_path.find("../../reference_spectra/")==0:					
						if sys.platform == "win32":
							mpath = path.replace("packages/GUI/GUI_SpectralAnalysis.py","reference_spectra/") + spectra_path.replace("../../reference_spectra/","")
						else:
							mpath = path.replace("packages/GUI/GUI_SpectralAnalysis.py","reference_spectra/") + spectra_path.replace("../../reference_spectra/","")						
						if os.path.isfile(mpath) == False:
							writeLog(self,"Row: " + str(csv_row) + " [Reference_Spectrum_Product]: "+spectra_path.replace("../../reference_spectra/","")+" cant be found in 'reference_spectra'-folder.")
							errorcount += 1
						else:
							rs_exist = False
							for rs_row in range(len(ref_spectra_lst)):
								if ref_spectra_lst[rs_row][0] == mpath:
									rs_exist = True
									break
							if rs_exist == False:					
								ref_spectra = pd.read_csv(mpath, skip_blank_lines=True).dropna(how="all")
								ref_spectra.index += 1
								ref_spectra_lst.append([mpath,ref_spectra.Wavelength[1],ref_spectra.Wavelength[len(ref_spectra.index)]])
								rs_row = len(ref_spectra_lst)-1
							if df.Wavelength_Start[row] < ref_spectra_lst[rs_row][1]:
								writeLog(self,"Row: " + str(csv_row) + " [Reference_Spectrum_Product]: "+spectra_path.replace("../../reference_spectra/","")+" Wavelength starts at " +str(ref_spectra_lst[rs_row][1])+" instead of "+ str(df.Wavelength_Start[row]))
								errorcount += 1								
							if df.Wavelength_End[row] > ref_spectra_lst[rs_row][2]:
								writeLog(self,"Row: " + str(csv_row) + " [Reference_Spectrum_Product]: "+spectra_path.replace("../../reference_spectra/","")+" Wavelength ends at " +str(ref_spectra_lst[rs_row][2])+" instead of " + str(df.Wavelength_End[row]))
								errorcount += 1							
					else:
						writeLog(self,"Row: " + str(csv_row) + " [Reference_Spectrum_Product] has to start with: ../../reference_spectra/")
						errorcount += 1
				if pd.isna(df.Total_Concentration[row]):
					writeLog(self,"Row: " + str(csv_row) + " [Total_Concentration] is missing.")
					errorcount += 1	
		count += 1
		self.ui.progressBar.setValue(count)
		
	writeLog(self,"\nGeneral check completed["+str(round(time.time()-start_time,2))+"s]")
	
	# Check Timepoint has to be unique per Condition_Name
	start_time_TP = time.time()
	tr1 = df[df.duplicated(["Condition_Name","Time_Point"],False)]
	for condition_name in tr1.Condition_Name.unique():
		tr2 = tr1[tr1["Condition_Name"] == condition_name]
		writeLog(self,"\n'" + str(condition_name) +"' has duplicate Timepoints:\n[Row] [Time_Point]\n"+str(tr2.Time_Point.to_string()))
		errorcount += 1	
	writeLog(self,"\nTime_Point check completed["+str(round(time.time()-start_time_TP,2))+"s]")
	self.ui.progressBar.setValue(maxcount-len(df.index))
	# Check Unique_Sample_ID
	start_time_US = time.time()
	tr1 = df[df.duplicated(["Unique_Sample_ID"],False)]
	if len(tr1)>0:
		writeLog(self,"\nThe following Sample_ID's are not unique:\n[Row] [Unique_Sample_ID]\n"+str(tr1.Unique_Sample_ID.to_string()))
		errorcount += len(tr1.index)
	writeLog(self,"\nUnique_Sample_ID check completed["+str(round(time.time()-start_time_US,2))+"s]")	
	self.ui.progressBar.setValue(maxcount)	
	writeLog(self,"\nmetadata-file check completed["+str(round(time.time()-start_time,2))+"s]:\n"+str(errorcount)+" Errors were found.\n")	
	
			
# initialize GUI and link controls with Formula
class MyApp(QMainWindow):
	def __init__(self):
		super(MyApp,self).__init__()
		self.ui= Ui_MainWindow()
		self.ui.setupUi(self)
		self.ui.btn_new.clicked.connect(self.New) # Connect Function with Button
		self.ui.btn_opendirectory.clicked.connect(self.OpenDirectory)
		self.ui.btn_selectrawfiles.clicked.connect(self.SelectRawFiles)	
		self.ui.btn_clearrawfiles.clicked.connect(self.ClearRawFiles)
		self.ui.btn_selectdestination.clicked.connect(self.SelectDestination)
		self.ui.btn_gensplit.clicked.connect(self.GenSplit)		
		self.ui.btn_selectmetafile.clicked.connect(self.SelectMetaFile)
		self.ui.btn_openmeta.clicked.connect(self.OpenMetafile)
		self.ui.btn_checkmetadata.clicked.connect(self.CheckMetadata)
		self.ui.btn_datatoolbox.clicked.connect(self.DataToolbox)
		self.ui.btn_checkfiles.clicked.connect(self.CheckFiles)
		self.ui.btn_clearlog.clicked.connect(self.ClearLog)		
		self.ui.btn_gensplit.setEnabled(False)
		self.ui.btn_selectmetafile.setEnabled(False)
		self.ui.btn_openmeta.setEnabled(False)
		self.ui.btn_checkmetadata.setEnabled(False)
		self.ui.btn_datatoolbox.setEnabled(False)
		self.ui.cb_gensplit.setChecked(False)
		self.ui.cb_metadata.setChecked(False)
		self.ui.cb_datatoolbox.setChecked(False)
		self.ui.progressBar.setValue(0)

# Reset the whole GUI
	def New(self):
		self.ui.lbl_te_rawfiles.setText("")
		self.ui.lbl_le_destinationfolder.setText("")
		self.ui.lbl_le_metafile.setText("")
		checkpath(self)
		self.ui.lbl_te_log.setText("")

# Open the Working Directory
	def OpenDirectory(self):
		path = os.path.realpath(__file__) 
		if sys.platform == "win32":
			path = path.replace("packages/GUI/GUI_SpectralAnalysis.py","")		
			os.startfile(path)
		else:
			path = path.replace("packages/GUI/GUI_SpectralAnalysis.py","")
			opener = "open" if sys.platform == "darwin" else "xdg-open"
			subprocess.call([opener, path])


	def SelectRawFiles(self):
		path = os.path.realpath(__file__) 
		if sys.platform == "win32":
			path = path.replace("packages/GUI/GUI_SpectralAnalysis.py","raw_data")
			fname = QFileDialog.getOpenFileNames(self, 'Select file(s)',path,"Text files (*.txt)") 	# returns a tuple ('path','filetype')			
		else:
			path = path.replace("packages/GUI/GUI_SpectralAnalysis.py","raw_data")
			fname = QFileDialog.getOpenFileNames(self, 'Select file(s)',path,"Text files (*.txt)",options=QFileDialog.DontUseNativeDialog) 	# returns a tuple ('path','filetype')
		path = ""
		if not fname[0]:
			writeLog(self,"Selection canceled.")
		else:
			if self.ui.lbl_te_rawfiles.toPlainText()  == "":
				#Autofill destination-folder
				filepath = fname[0][0]
				file = filepath.split("/")[-1]
				self.ui.lbl_le_destinationfolder.setText(filepath.replace("raw_data","processed_data").replace(file,""))
				writeLog(self,"Destination selected.")
				#Autofill metafile
				if os.path.isfile(filepath.replace("raw_data","metadata").replace("/" + file,".csv"))== True:
					self.ui.lbl_le_metafile.setText(filepath.replace("raw_data","metadata").replace("/" + file,".csv"))
					writeLog(self,"Metafile selected.")
			for filepath in fname[0]:
				self.ui.lbl_te_rawfiles.append(filepath)
			writeLog(self,"Platereader-files selected.")
			checkpath(self)

	def ClearRawFiles(self):
		self.ui.lbl_te_rawfiles.setText("")
		writeLog(self,"clear rawfiles")
		checkpath(self)
			
	def SelectDestination(self):
		path = os.path.realpath(__file__) 
		if sys.platform == "win32":
			path = path.replace("packages/GUI/GUI_SpectralAnalysis.py","processed_data")
			fname = QFileDialog.getExistingDirectory(self, 'Select folder',path) 	# returns a tuple ('path','filetype')			
		else:
			path = path.replace("packages/GUI/GUI_SpectralAnalysis.py","processed_data")		
			fname = QFileDialog.getExistingDirectory(self, 'Select folder',path,options=QFileDialog.DontUseNativeDialog) 	# returns a tuple ('path','filetype')
		if not fname:
			writeLog(self,"Selection canceled.")
		else:
			self.ui.lbl_le_destinationfolder.setText(fname)	
			writeLog(self,"Destination selected.")
			if os.path.isfile(fname.replace("processed_data","metadata")+".csv")== True:
				self.ui.lbl_le_metafile.setText(fname.replace("processed_data","metadata")+".csv")
				writeLog(self,"Metafile selected.")
			checkpath(self)
			
	def GenSplit(self):
		disableButtons(self)
		rawpath = self.ui.lbl_te_rawfiles.toPlainText()
		#progress Bar		
		maxcount = round(len(rawpath.splitlines()))
		count = 0
		self.ui.progressBar.setMaximum(maxcount)
		self.ui.progressBar.setValue(count)
		processedpath = self.ui.lbl_le_destinationfolder.text()
		os.makedirs(processedpath, exist_ok=True)
		# go through each file
		writeLog(self,"\nGen5Split processing...")
		for rawfile in rawpath.splitlines():
			#Call Gen5Split
			print(rawfile)
			processGenSplit(self,rawfile,processedpath)
			count +=1
			self.ui.progressBar.setValue(count)		
		writeLog(self,"Gen5split finished.\n")
		checkpath(self)
			
	def SelectMetaFile(self):
		path = os.path.realpath(__file__) 
		if sys.platform == "win32":
			path = path.replace("packages/GUI/GUI_SpectralAnalysis.py","metadata")
			fname = QFileDialog.getOpenFileName(self, 'Select Template File',path,"CSV (*.csv)") 	# returns a tuple ('list(paths)','filetype')			
		else:
			path = path.replace("packages/GUI/GUI_SpectralAnalysis.py","metadata")	
			fname = QFileDialog.getOpenFileName(self, 'Select Template File',path,"CSV (*.csv)",options=QFileDialog.DontUseNativeDialog) 	# returns a tuple ('list(paths)','filetype')
		if not fname[0]:
			writeLog(self,"Selection canceled.")	
		else:
			metafile = fname[0]
			self.ui.lbl_le_metafile.setText(metafile)
			writeLog(self,"Metafile selected.")
			checkpath(self)			
			
	def OpenMetafile(self):
		metafile = self.ui.lbl_le_metafile.text()
		#Open File (system standard)
		p = Popen(metafile, shell=True)	

	def CheckMetadata(self):
		checkMetadata(self,self.ui.lbl_le_metafile.text())
			
	def DataToolbox(self):
		disableButtons(self)
		#get the name of the metafile and the processed_data-path
		processedpath = self.ui.lbl_le_destinationfolder.text()
		metafile = self.ui.lbl_le_metafile.text()		
		#progress Bar
		df = pd.read_csv(metafile, skip_blank_lines=True,usecols = ["Condition_Name"]).dropna(how="all")
		maxcount = (len(df.Condition_Name.unique())+2)*14
		count = 0
		self.ui.progressBar.setMaximum(maxcount)
		self.ui.progressBar.setValue(count)
		
		start_time = time.time()
		#call datatoolbox
		writeLog(self,"\nDataToolbox processing. Please wait.")
		process = subprocess.Popen(["data_toolbox", "-m",metafile], stdout=subprocess.PIPE,stderr=subprocess.PIPE, cwd=processedpath)
		#Create Log (realtime, line by line)
		while True:
			output = process.stderr.readline()
			if output == b'' and process.poll() is not None:
				break
			if output:
				output_str=str(output.strip())
				print(output_str)
				writeLog(self,output_str.replace("b'","").replace("'","").replace(r"\r\n","").replace("b\"",""))
				#progressbar
				if output_str.find("I am plotting now condition")!=-1:
					count+=1
					self.ui.progressBar.setValue(count)
				elif output_str.find("Fitting condition")!=-1:
					count+=13
					self.ui.progressBar.setValue(count)
					print("Remaining Time:" +str(round((time.time()-start_time)*(maxcount-count)/count,2))+"s / "+str(round((time.time()-start_time)*maxcount/count,2)) +"s")
				
		if output_str.find("Done!") != -1:
			self.ui.progressBar.setValue(maxcount)	
			writeLog(self,"\nDataToolbox finished.["+str(round(time.time()-start_time,2))+"s]\n")
		else:
			writeLog(self,"\nDataToolbox failed. Try 'Check metadata file'!\n")			
		checkpath(self)
			
	def CheckFiles(self):
		checkpath(self)
		
	def ClearLog(self):
		self.ui.lbl_te_log.setText("")

if __name__ == "__main__":
	app = QApplication(sys.argv)
	window = MyApp()
	window.show()
	sys.exit(app.exec_())


