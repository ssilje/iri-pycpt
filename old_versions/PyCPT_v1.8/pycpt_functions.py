#This is PyCPT_functions.py (version1.8) -- 18 Sep 2020
#Authors: ÁG Muñoz (agmunoz@iri.columbia.edu), AW Robertson (awr@iri.columbia.edu), T Turkington (NEA), Bohar Singh, SJ Mason
#Notes: be sure it matches version of PyCPT
#Log: see version.log in GitHub
# AWR: edits made 20-Mar-2020: (1) indentation errors fixed; (2) ‘mpref’ argument to Prepfiles & GetForecast (L131, L143, L166, L1648). Argument added to calling program too (PyCPT_s2sv1.6.ipynb); (3) added mpref if-statement to GetForecasts (L1672) to only get the individual ensemble members for noMOS; (4) Revised dictionary entry for CFSv2 *forecast* file from S2S database (L1659)


import os
import warnings
import struct
import xarray as xr
import numpy as np
import pandas as pd
from copy import copy
from scipy.stats import t
from scipy.stats import invgamma
import cartopy.crs as ccrs
from cartopy import feature
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from netCDF4 import Dataset


warnings.filterwarnings("ignore")

def lines_that_equal(line_to_match, fp):
	return [line for line in fp if line == line_to_match]

def lines_that_contain(string, fp):
	return [line for line in fp if string in line]

def lines_that_start_with(string, fp):
	return [line for line in fp if line.startswith(string)]

def lines_that_end_with(string, fp):
	return [line for line in fp if line.endswith(string)]

def exceedprob(x,dof,lo,sc):
	return t.sf(x, dof, loc=lo, scale=sc)*100

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def discrete_cmap(N, base_cmap=None):
	"""Create an N-bin discrete colormap from the specified input map"""
	# Note that if base_cmap is a string or None, you can simply do
	#    return plt.cm.get_cmap(base_cmap, N)
	# The following works for string, None, or a colormap instance:
	base = plt.cm.get_cmap(base_cmap)
	color_list = base(np.linspace(0, 1, N))
	cmap_name = base.name + str(N)
	#base.set_bad(color='white')
	#return base.from_list(cmap_name, color_list, N)
	return LinearSegmentedColormap.from_list(cmap_name, color_list, N) #perceptually uniform colormaps

def ncdump(nc_fid, verb=True):
    '''
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.
	Note: Modified by Ángel G. Muñoz from original version by Chris Slocum - CSU.

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
    verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
    '''
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print ("\t\ttype:", repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print ('\t\t%s:' % ncattr,\
                      repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print ("\t\tWARNING: %s does not contain variable attributes" % key)

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print ("NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print ('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print ("NetCDF dimension information:")
        for dim in nc_dims:
            print ("\tName:", dim)
            print ("\t\tsize:", len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print ("NetCDF variable information:")
        for var in nc_vars:
            if var not in nc_dims:
                print ('\tName:', var)
                print ("\t\tdimensions:", nc_fid.variables[var].dimensions)
                print ("\t\tsize:", nc_fid.variables[var].size)
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars

def PrepFiles(rainfall_frequency, threshold_pctle, wlo1, wlo2,elo1, elo2,sla1, sla2,nla1, nla2, day1, day2, fday, nday, fyr, mon, os, authkey, wk, wetday_threshold, nlag, training_season, hstep, model, obs_source, obsclimo_source, hdate_last, force_download,mpref):
	"""Function to download (or not) the needed files"""
	if obs_source=='userdef':
		GetHindcastsUser(wlo1, elo1, sla1, nla1, day1, day2, fyr, mon, os, authkey, wk, nlag, nday, training_season, hstep, model, hdate_last, force_download)
		print('Hindcasts file ready to go')
		print('----------------------------------------------')
		GetObsUser(day1, day2, mon, fyr, wlo2, elo2, sla2, nla2, nday, authkey, wk, nlag, training_season, hstep, model, obs_source, obsclimo_source, hdate_last, force_download)
		print('Obs:precip file ready to go')
		print('----------------------------------------------')
		GetForecastUser(day1, day2, fday, mon, fyr, nday, wlo1, elo1, sla1, nla1, wlo2, elo2, sla2, nla2, obs_source, authkey, wk, nlag, model, hdate_last, threshold_pctle,training_season,wetday_threshold,force_download,mpref)
		print('Forecasts file ready to go')
		print('----------------------------------------------')
	else:
		if rainfall_frequency:
			GetObs_RFREQ(day1, day2, mon, fyr, wlo2, elo2, sla2, nla2, nday, authkey, wk, wetday_threshold, threshold_pctle, nlag, training_season, hstep, model, obs_source, force_download)
			print('Obs:rfreq file ready to go')
			print('----------------------------------------------')
#			nday added after nlag for GEFS & CFSv2
			GetHindcasts(wlo1, elo1, sla1, nla1, day1, day2, fyr, mon, os, authkey, wk, nlag, nday, training_season, hstep, model, hdate_last, force_download)
			#GetHindcasts_RFREQ(wlo1, elo1, sla1, nla1, day1, day2, nday, fyr, mon, os, authkey, wk, wetday_threshold, nlag, training_season, hstep, model, force_download)
			print('Hindcasts file ready to go')
			print('----------------------------------------------')
			#GetForecast_RFREQ(day1, day2, fday, mon, fyr, nday, wlo1, elo1, sla1, nla1, authkey, wk, wetday_threshold, nlag, model, force_download)
			GetForecast(day1, day2, fday, mon, fyr, nday, wlo1, elo1, sla1, nla1, wlo2, elo2, sla2, nla2, obs_source, authkey, wk, nlag, model, hdate_last,threshold_pctle,training_season,wetday_threshold,force_download,mpref)
			print('Forecasts file ready to go')
			print('----------------------------------------------')
		else:
			# if temp:
			# 	GetHindcasts_Temp(wlo1, elo1, sla1, nla1, day1, day2, fyr, mon, os, key, week, nlag, nday, training_season, hstep, model, hdate_last, force_download)
			# 	print('Hindcasts file ready to go')
			# 	print('----------------------------------------------')
			# 	GetObsTn(day1, day2, mon, fyr, wlo2, elo2, sla2, nla2, nday, key, week, nlag, training_season, hstep, model, obs_source, hdate_last, force_download)
			# 	print('Obs:temp min file ready to go')
			# 	print('----------------------------------------------')
			# 	GetForecast_Temp(day1, day2, fday, mon, fyr, nday, wlo1, elo1, sla1, nla1, wlo2, elo2, sla2, nla2, obs_source, key, week, nlag, model, hdate_last, threshold_pctle,training_season,wetday_threshold,force_download)
			# 	print('Forecasts file ready to go')
			# 	print('----------------------------------------------')
			# else:
			#GetHindcasts(wlo1, elo1, sla1, nla1, day1, day2, fyr, mon, os, authkey, wk, nlag, training_season, hstep, model, force_download)
			#nday added after nlag for GEFS & CFSv2
			GetHindcasts(wlo1, elo1, sla1, nla1, day1, day2, fyr, mon, os, authkey, wk, nlag, nday, training_season, hstep, model, hdate_last, force_download)
			print('Hindcasts file ready to go')
			print('----------------------------------------------')
			GetObs(day1, day2, mon, fyr, wlo2, elo2, sla2, nla2, nday, authkey, wk, nlag, training_season, hstep, model, obs_source, obsclimo_source, hdate_last, force_download)
			print('Obs:precip file ready to go')
			print('----------------------------------------------')
			GetForecast(day1, day2, fday, mon, fyr, nday, wlo1, elo1, sla1, nla1, wlo2, elo2, sla2, nla2, obs_source,obsclimo_source, authkey, wk, nlag, model, hdate_last, threshold_pctle,training_season,wetday_threshold,force_download,mpref)
			print('Forecasts file ready to go')
			print('----------------------------------------------')

def PrepFiles_usrNetcdf(fprefix, predictand, wlo1, wlo2,elo1, elo2, sla1, sla2, nla1, nla2, tgti, tgtf, mon, monf, fyr, tar, infile_predictand, infile_hindcast, infile_forecast):
		"""Function to user-provided NetCDF files"""

		readNetCDF_predictand(infile_predictand,outfile, predictand, wlo2, elo2, sla2, nla2, tar)
		print('Obs:precip file ready to go')
		print('----------------------------------------------')

		readNetCDF_Hindcasts(infile_hindcast, outfile, wlo1, elo1, sla1, nla1, tgti, tgtf, mon, tar)
		print('Hindcasts file ready to go')
		print('----------------------------------------------')

		readNetCDF_Forecast(infile_forecast, outfile, monf, fyr, tgti, tgtf, tar, wlo1, elo1, sla1, nla1)
		print('Forecasts file ready to go')
		print('----------------------------------------------')

def pltdomain(loni1,lone1,lati1,late1,loni2,lone2,lati2,late2):
	"""A simple plot function for the geographical domain

	PARAMETERS
	----------
		loni: western longitude
		lone: eastern longitude
		lati: southern latitude
		late: northern latitude
		title: title
	"""
	#Create a feature for States/Admin 1 regions at 1:10m from Natural Earth
	states_provinces = feature.NaturalEarthFeature(
		category='cultural',
		name='admin_1_states_provinces_shp',
		scale='10m',
		facecolor='none')

	fig = plt.subplots(figsize=(15,15), subplot_kw=dict(projection=ccrs.PlateCarree()))
	loni = [loni1,loni2]
	lati = [lati1,lati2]
	lone = [lone1,lone2]
	late = [late1,late2]
	title = ['Predictor', 'Predictand']

	for i in range(2):

		ax = plt.subplot(1, 2, i+1, projection=ccrs.PlateCarree())
		ax.set_extent([loni[i],lone[i],lati[i],late[i]], ccrs.PlateCarree())

		# Put a background image on for nice sea rendering.
		ax.stock_img()

		ax.add_feature(feature.LAND)
		ax.add_feature(feature.COASTLINE)
		ax.set_title(title[i]+" domain")
		pl=ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
				  linewidth=2, color='gray', alpha=0.5, linestyle='--')
		pl.xlabels_top = False
		pl.ylabels_left = False
		pl.xformatter = LONGITUDE_FORMATTER
		pl.yformatter = LATITUDE_FORMATTER
		ax.add_feature(states_provinces, edgecolor='gray')
	plt.show()

def pltmap(score,loni,lone,lati,late,fprefix,mpref,training_season, mon, fday, nwk, wki):
# wki is the week identifier, eg 1,2,34

	"""A simple function for ploting the statistical score

	PARAMETERS
	----------
		score: the score
		loni: western longitude
		lone: eastern longitude
		lati: southern latitude
		late: northern latitude
		title: title
	"""

	plt.figure(figsize=(20,5))

	for L in range(nwk):
		wk=L+1
		#Read grads binary file size H, W  --it assumes all files have the same size, and that 2AFC exists
		with open('../output/'+fprefix+'_'+mpref+'_2AFC_'+training_season+'_wk'+str(wki[wk-1])+'.ctl', "r") as fp:
			for line in lines_that_contain("XDEF", fp):
				W = int(line.split()[1])
				XD= float(line.split()[4])
		with open('../output/'+fprefix+'_'+mpref+'_2AFC_'+training_season+'_wk'+str(wki[wk-1])+'.ctl', "r") as fp:
			for line in lines_that_contain("YDEF", fp):
				H = int(line.split()[1])
				YD= float(line.split()[4])

#		ax = plt.subplot(nwk/2, 2, wk, projection=ccrs.PlateCarree())
		ax = plt.subplot(1,nwk, wk, projection=ccrs.PlateCarree())
		ax.set_extent([loni,loni+W*XD,lati,lati+H*YD], ccrs.PlateCarree())

		#Create a feature for States/Admin 1 regions at 1:10m from Natural Earth
		states_provinces = feature.NaturalEarthFeature(
			category='cultural',
#			name='admin_1_states_provinces_shp',
			name='admin_0_countries',
			scale='10m',
			facecolor='none')

		ax.add_feature(feature.LAND)
		ax.add_feature(feature.COASTLINE)
		ax.set_title(score+' for Week '+str(wki[wk-1]))
		pl=ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
				  linewidth=2, color='gray', alpha=0., linestyle='--')
		pl.xlabels_top = False
		pl.ylabels_left = True
		pl.ylabels_right = False
		pl.xformatter = LONGITUDE_FORMATTER
		pl.yformatter = LATITUDE_FORMATTER
		ax.add_feature(states_provinces, edgecolor='gray')
		lon_formatter = LongitudeFormatter(degree_symbol='')
		lat_formatter = LatitudeFormatter(degree_symbol='')
		ax.xaxis.set_major_formatter(lon_formatter)
		ax.yaxis.set_major_formatter(lat_formatter)
		ax.set_ybound(lower=lati, upper=late)

		if score == 'CCAFCST_V' or score == 'PCRFCST_V' or score == 'noMOSFCST_V':
			f=open('../output/'+fprefix+'_'+score+'_'+training_season+'_'+mon+str(fday)+'_wk'+str(wki[wk-1])+'.dat','rb')
			recl=struct.unpack('i',f.read(4))[0]
			numval=int(recl/np.dtype('float32').itemsize)
			#Now we read the field
			A=np.fromfile(f,dtype='float32',count=numval)
			var = np.transpose(A.reshape((W, H), order='F'))
			var[var==-999.]=np.nan #only sensible values
			current_cmap = plt.cm.BrBG
			current_cmap.set_bad('white',1.0)
			current_cmap.set_under('white', 1.0)
			if fprefix == 'RFREQ':
				label ='Freq Rainy Days (days)'
				var=var/100 #weird 100 factor coming from CPT for frq rainy days!! ??
			elif fprefix == 'PRCP':
				label = 'Rainfall anomaly (mm/week)'
			CS=plt.pcolormesh(np.linspace(loni, loni+W*XD,num=W), np.linspace(lati+H*YD, lati, num=H), var,
				#vmin=-max(np.max(var),np.abs(np.min(var))), #vmax=np.max(var),
                vmin=-max(np.nanmax(var),np.abs(np.nanmin(var))),
				vmax= max(np.nanmax(var),np.abs(np.nanmin(var))),
				#norm=MidpointNormalize(midpoint=0.),
				cmap=current_cmap,
				transform=ccrs.PlateCarree())
			ax.set_title("Deterministic forecast for Week "+str(wki[wk-1]))
			f.close()
			#current_cmap = plt.cm.get_cmap()
			#current_cmap.set_bad(color='white')
			#current_cmap.set_under('white', 1.0)
		else:
			#Since CPT writes grads files in sequential format, we need to excise the 4 bytes between records (recl)
			f=open('../output/'+fprefix+'_'+mpref+'_'+score+'_'+training_season+'_wk'+str(wki[wk-1])+'.dat','rb')
			recl=struct.unpack('i',f.read(4))[0]
			numval=int(recl/np.dtype('float32').itemsize)
			#Now we read the field
			A=np.fromfile(f,dtype='float32',count=numval)
			var = np.transpose(A.reshape((W, H), order='F'))
			#define colorbars, depending on each score	--This can be easily written as a function
			if score == '2AFC':
				var[var<0]=np.nan #only positive values
				CS=plt.pcolormesh(np.linspace(loni, loni+W*XD,num=W), np.linspace(lati+H*YD, lati, num=H), var,
				vmin=0,vmax=100,
				cmap=discrete_cmap(11, 'bwr'),
				transform=ccrs.PlateCarree())
				label = '2AFC (%)'

			if score == 'RocAbove' or score=='RocBelow':
				var[var<0]=np.nan #only positive values
				CS=plt.pcolormesh(np.linspace(loni, loni+W*XD,num=W), np.linspace(lati+H*YD, lati, num=H), var,
				vmin=0,vmax=1,
				cmap=discrete_cmap(11, 'bwr'),
				transform=ccrs.PlateCarree())
				label = 'ROC area'

			if score == 'Spearman' or score=='Pearson':
				var[var<-1.]=np.nan #only sensible values
				CS=plt.pcolormesh(np.linspace(loni, loni+W*XD,num=W), np.linspace(lati+H*YD, lati, num=H), var,
				vmin=-1,vmax=1,
				cmap=discrete_cmap(11, 'bwr'),
				transform=ccrs.PlateCarree())
				label = 'Correlation'

			if score == 'RPSS':
				var[var==-999.]=np.nan #only sensible values
				CS=plt.pcolormesh(np.linspace(loni, loni+W*XD,num=W), np.linspace(lati+H*YD, lati, num=H), var,
				vmin=-20,vmax=20,
				cmap=discrete_cmap(20, 'bwr'),
				transform=ccrs.PlateCarree())
				label = 'RPSS (all categories)'

			if score=='GROC':
				var[var<-1.]=np.nan #only sensible values
				CS=plt.pcolormesh(np.linspace(loni, loni+W*XD,num=W), np.linspace(lati+H*YD, lati, num=H), var,
				vmin=0,vmax=100,
				cmap=discrete_cmap(11, 'bwr'),
				transform=ccrs.PlateCarree())
				label = 'GROC (probabilistic)'

			if score=='Ignorance':
				var[var<-1.]=np.nan #only sensible values
				CS=plt.pcolormesh(np.linspace(loni, loni+W*XD,num=W), np.linspace(lati+H*YD, lati, num=H), var,
				#vmin=1.,vmax=2.,
				cmap=discrete_cmap(20, 'bwr'),
				transform=ccrs.PlateCarree())
				label = 'Ignorance (all categories)'
		f.close()
	plt.subplots_adjust(hspace=0)
	plt.subplots_adjust(bottom=0.20, top=0.9)
	cax  = plt.axes([0.2, 0.08, 0.6, 0.04])
	cbar = plt.colorbar(CS,cax=cax, orientation='horizontal')
	cbar.set_label(label) #, rotation=270)

def pltmapdiff(score,loni,lone,lati,late,fprefix,mpref1,mpref2,training_season, mon, fday, nwk):
	"""A simple function for ploting differences of the skill scores

	PARAMETERS
	----------
		score: the score
		loni: western longitude
		lone: eastern longitude
		lati: southern latitude
		late: northern latitude
		title: title
	"""

	plt.figure(figsize=(20,5))

	for L in range(nwk):
		wk=L+1
		#Read grads binary file size H, W  --it assumes all files have the same size, and that 2AFC exists
		with open('../output/'+fprefix+'_'+mpref1+'_2AFC_'+training_season+'_wk'+str(wk)+'.ctl', "r") as fp:
			for line in lines_that_contain("XDEF", fp):
				W = int(line.split()[1])
				XD= float(line.split()[4])
		with open('../output/'+fprefix+'_'+mpref1+'_2AFC_'+training_season+'_wk'+str(wk)+'.ctl', "r") as fp:
			for line in lines_that_contain("YDEF", fp):
				H = int(line.split()[1])
				YD= float(line.split()[4])

#		ax = plt.subplot(nwk/2, 2, wk, projection=ccrs.PlateCarree())
		ax = plt.subplot(1,nwk, wk, projection=ccrs.PlateCarree())
		ax.set_extent([loni,loni+W*XD,lati,lati+H*YD], ccrs.PlateCarree())

		#Create a feature for States/Admin 1 regions at 1:10m from Natural Earth
		states_provinces = feature.NaturalEarthFeature(
			category='cultural',
#			name='admin_1_states_provinces_shp',
			name='admin_0_countries',
			scale='10m',
			facecolor='none')

		ax.add_feature(feature.LAND)
		ax.add_feature(feature.COASTLINE)
		ax.set_title('Week '+str(wk))
		pl=ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
				  linewidth=2, color='gray', alpha=0., linestyle='--')
		pl.xlabels_top = False
		pl.ylabels_left = True
		pl.ylabels_right = False
		pl.xformatter = LONGITUDE_FORMATTER
		pl.yformatter = LATITUDE_FORMATTER
		ax.add_feature(states_provinces, edgecolor='gray')
		ax.set_ybound(lower=lati, upper=late)

		if score == 'CCAFCST_V' or score == 'PCRFCST_V' or score == 'noMOSFCST_V':
			f=open('../output/'+fprefix+'_'+score+'_'+training_season+'_'+mon+str(fday)+'_wk'+str(wk)+'.dat','rb')
			recl=struct.unpack('i',f.read(4))[0]
			numval=int(recl/np.dtype('float32').itemsize)
			#Now we read the field
			A=np.fromfile(f,dtype='float32',count=numval)
			var = np.transpose(A.reshape((W, H), order='F'))
			var[var==-999.]=np.nan #only sensible values
			current_cmap = plt.cm.BrBG
			current_cmap.set_bad('white',1.0)
			current_cmap.set_under('white', 1.0)
			if fprefix == 'RFREQ':
				label ='Freq Rainy Days (days)'
				var=var/100 #weird 100 factor coming from CPT for frq rainy days!! ??
			elif fprefix == 'PRCP':
				label = 'Rainfall anomaly (mm/week)'
			CS=plt.pcolormesh(np.linspace(loni, loni+W*XD,num=W), np.linspace(lati+H*YD, lati, num=H), var,
				#vmin=-max(np.max(var),np.abs(np.min(var))), #vmax=np.max(var),
				norm=MidpointNormalize(midpoint=0.),
				cmap=current_cmap,
				transform=ccrs.PlateCarree())
			ax.set_title("Deterministic forecast for Week "+str(wk))
			f.close()
			#current_cmap = plt.cm.get_cmap()
			#current_cmap.set_bad(color='white')
			#current_cmap.set_under('white', 1.0)
		else:
			#Since CPT writes grads files in sequential format, we need to excise the 4 bytes between records (recl)
			f=open('../output/'+fprefix+'_'+mpref1+'_'+score+'_'+training_season+'_wk'+str(wk)+'.dat','rb')
			recl=struct.unpack('i',f.read(4))[0]
			numval=int(recl/np.dtype('float32').itemsize)
			#Now we read the field
			A=np.fromfile(f,dtype='float32',count=numval)
			var1 = np.transpose(A.reshape((W, H), order='F'))
			var1[var1==-999]=np.nan
			f=open('../output/'+fprefix+'_'+mpref2+'_'+score+'_'+training_season+'_wk'+str(wk)+'.dat','rb')
			recl=struct.unpack('i',f.read(4))[0]
			numval=int(recl/np.dtype('float32').itemsize)
			#Now we read the field
			A=np.fromfile(f,dtype='float32',count=numval)
			var2 = np.transpose(A.reshape((W, H), order='F'))
			var2[var2==-999]=np.nan
			var=var2-var1
			vmi=-max(np.nanmax(var),np.abs(np.nanmin(var)))
			vma=-vmi
			#define colorbars, depending on each score	--This can be easily written as a function
			if score == '2AFC':
				CS=plt.pcolormesh(np.linspace(loni, loni+W*XD,num=W), np.linspace(lati+H*YD, lati, num=H), var,
				vmin=vmi, vmax=vma,
				cmap=discrete_cmap(11, 'bwr'),
				transform=ccrs.PlateCarree())
				label = '2AFC (%)'
			if score == 'RocAbove' or score=='RocBelow':
				CS=plt.pcolormesh(np.linspace(loni, loni+W*XD,num=W), np.linspace(lati+H*YD, lati, num=H), var,
				vmin=vmi, vmax=vma,
				cmap=discrete_cmap(11, 'bwr'),
				transform=ccrs.PlateCarree())
				label = 'ROC area'
			if score == 'Spearman' or score=='Pearson':
				CS=plt.pcolormesh(np.linspace(loni, loni+W*XD,num=W), np.linspace(lati+H*YD, lati, num=H), var,
				vmin=vmi, vmax=vma,
				cmap=discrete_cmap(11, 'bwr'),
				transform=ccrs.PlateCarree())
				label = 'Correlation'
			if score == 'RPSS':
				CS=plt.pcolormesh(np.linspace(loni, loni+W*XD,num=W), np.linspace(lati+H*YD, lati, num=H), var,
				vmin=vmi, vmax=vma,
				cmap=discrete_cmap(20, 'bwr'),
				transform=ccrs.PlateCarree())
				label = 'RPSS (all categories)'
			if score=='GROC':
				CS=plt.pcolormesh(np.linspace(loni, loni+W*XD,num=W), np.linspace(lati+H*YD, lati, num=H), var,
				vmin=vmi, vmax=vma,
				cmap=discrete_cmap(11, 'bwr'),
				transform=ccrs.PlateCarree())
				label = 'GROC (probabilistic)'
			if score=='Ignorance':
				CS=plt.pcolormesh(np.linspace(loni, loni+W*XD,num=W), np.linspace(lati+H*YD, lati, num=H), var,
				vmin=vmi, vmax=vma,
				cmap=discrete_cmap(20, 'bwr'),
				transform=ccrs.PlateCarree())
				label = 'Ignorance (all categories)'

		plt.subplots_adjust(hspace=0)
		#plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
		#cbar_ax = plt.add_axes([0.85, 0.15, 0.05, 0.7])
		#plt.tight_layout()
		plt.subplots_adjust(bottom=0.15, top=0.9)
		cax = plt.axes([0.2, 0.08, 0.6, 0.04])
		cbar = plt.colorbar(CS,cax=cax, orientation='horizontal')
		cbar.set_label(label) #, rotation=270)
		f.close()


def skilltab(score,wknam,lon1,lat1,lat2,lon2,loni,lone,lati,late,fprefix,mpref,training_season,mon,fday,nwk,wki):
	"""Creates a table with min, max and average values of skills computed over a certain domain

	PARAMETERS
	----------
		thrs: the threshold, in the units of the predictand
		lon: longitude
		lat: latitude
	"""

	#Read grads binary file size H, W  --it assumes all files have the same size, and that 2AFC exists
	with open('../output/'+fprefix+'_'+mpref+'_2AFC_'+training_season+'_wk1.ctl', "r") as fp:
		for line in lines_that_contain("XDEF", fp):
			W = int(line.split()[1])
			XD= float(line.split()[4])
	with open('../output/'+fprefix+'_'+mpref+'_2AFC_'+training_season+'_wk1.ctl', "r") as fp:
		for line in lines_that_contain("YDEF", fp):
			H = int(line.split()[1])
			YD= float(line.split()[4])

	#Find the gridbox:
	lonrange = np.linspace(loni, loni+(W-1)*XD,num=W)
	latrange = np.linspace(lati+(H-1)*YD, lati, num=H)  #need to reverse the latitudes because of CPT (GrADS YREV option)
	lon_grid, lat_grid = np.meshgrid(lonrange, latrange)
	#first point
	a = abs(lat_grid-lat1)+abs(lon_grid-lon1)
	i1,j1 = np.unravel_index(a.argmin(),a.shape)   #i:latitude   j:longitude
	#second point
	a = abs(lat_grid-lat2)+abs(lon_grid-lon2)
	i2,j2 = np.unravel_index(a.argmin(),a.shape)   #i:latitude   j:longitude

	df = pd.DataFrame(index=wknam[0:nwk])
	for L in range(nwk):
		wk=L+1
		for S in score:
			#Since CPT writes grads files in sequential format, we need to excise the 4 bytes between records (recl)
			f=open('../output/'+fprefix+'_'+mpref+'_'+str(S)+'_'+training_season+'_wk'+str(wki[wk-1])+'.dat','rb')
			recl=struct.unpack('i',f.read(4))[0]
			numval=int(recl/np.dtype('float32').itemsize)
			#Now we read the field
			A=np.fromfile(f,dtype='float32',count=numval)
			var = np.transpose(A.reshape((W, H), order='F'))
			var[var==-999.]=np.nan #only sensible values
			df.at[wknam[L], str(S)] = round(np.nanmean(np.nanmean(var[i1:i2,j1:j2], axis=1), axis=0),2)
			df.at[wknam[L], 'max('+str(S)+')']  = round(np.nanmax(var[i1:i2,j1:j2]),2)
			df.at[wknam[L], 'min('+str(S)+')']  = round(np.nanmin(var[i1:i2,j1:j2]),2)
	return df
	f.close()

def pltmapProbNC(loni,lone,lati,late,fprefix,mpref,training_season, mon, fday,nwk,wki):
	"""A simple function for ploting probabilistic forecasts from netcdf files
	[FOR NOW IT ONLY WORKS FOR ECMWF]
	"""
	plt.figure(figsize=(15,15))
	#Create a feature for States/Admin 1 regions at 1:10m from Natural Earth
	states_provinces = feature.NaturalEarthFeature(
		category='cultural',
	#	name='admin_1_states_provinces_shp',
		name='admin_0_countries',
		scale='10m',
		facecolor='none')

	for L in range(nwk):
		wk=L+1

		#Read each tercile probabilities using the 3 different files downloaded from IRIDL
		nc_abo = Dataset('../input/noMOS/modelfcst_above_'+fprefix+'_'+mon+'_wk'+str(wki[wk-1])+'.nc', 'r')
		nc_bel = Dataset('../input/noMOS/modelfcst_below_'+fprefix+'_'+mon+'_wk'+str(wki[wk-1])+'.nc', 'r')
		nc_attrs, nc_dims, nc_vars = ncdump(nc_abo,verb=False)
		# Extract data from NetCDF file
		lats = nc_abo.variables['Y'][:]
		H = nc_abo.variables['Y'].size
		YD = 1.5 #ECMWF; in the future, read it from the Y:pointwidth attribute in the NC file
		lons = nc_abo.variables['X'][:]
		W = nc_abo.variables['X'].size
		XD = 1.5 #ECMWF; in the future, read it from the X:pointwidth attribute in the NC file
		probab = nc_abo.variables['flag'][:]
		probbe = nc_bel.variables['flag'][:]
		probno = [(x * 0.) + 100 for x in probab] - probab - probbe #we just compute the normal cat as the residual, to simplify things


		var=[probbe,probno,probab]

		tit=['Below Normal','Normal','Above Normal']
		for i in range(3):
			ax2=plt.subplot(nwk, 3, (L*3)+(i+1),projection=ccrs.PlateCarree())
			ax2.set_title("Week "+str(wki[wk-1])+ ": "+tit[i])
			ax2.add_feature(feature.LAND)
			ax2.add_feature(feature.COASTLINE)
			#ax2.set_ybound(lower=lati, upper=late)
			pl2=ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
				linewidth=2, color='gray', alpha=0.5, linestyle='--')
			pl2.xlabels_top = False
			pl2.ylabels_left = True
			pl2.ylabels_right = False
			pl2.xformatter = LONGITUDE_FORMATTER
			pl2.yformatter = LATITUDE_FORMATTER
			ax2.add_feature(states_provinces, edgecolor='black')
			ax2.set_extent([loni,loni+W*XD,lati,lati+H*YD], ccrs.PlateCarree())

			ax2.set_ybound(lower=lati, upper=late)
			ax2.set_xbound(lower=loni, upper=lone)
			#ax2.set_adjustable('box')
			#ax2.set_aspect('auto',adjustable='datalim',anchor='C')
			CS=ax2.pcolormesh(np.linspace(lons[0], lons[-1],num=W), np.linspace(lats[0], lats[-1], num=H), np.squeeze(var[i]),
			vmin=0,vmax=100,
			cmap=plt.cm.bwr,
			transform=ccrs.PlateCarree())
			#plt.show(block=False)

	plt.subplots_adjust(hspace=0)
	plt.subplots_adjust(bottom=0.15, top=0.9)
	cax = plt.axes([0.2, 0.08, 0.6, 0.04])
	cbar = plt.colorbar(CS,cax=cax, orientation='horizontal')
	cbar.set_label('Probability (%)') #, rotation=270)


def pltmapProb(loni,lone,lati,late,fprefix,mpref,training_season, mon, fday, nwk, wki):
	"""A simple function for ploting probabilistic forecasts

	PARAMETERS
	----------
		score: the score
		loni: western longitude
		lone: eastern longitude
		lati: southern latitude
		late: northern latitude
		title: title
	"""
	if mpref=='noMOS' and fprefix=='PRCP':
		pltmapProbNC(loni,lone,lati,late,fprefix,mpref,training_season, mon, fday, nwk,wki)
	else:
		#Need this score to be defined by the calibration method!!!
		score = mpref+'FCST_P'

		plt.figure(figsize=(15,15))
		#Create a feature for States/Admin 1 regions at 1:10m from Natural Earth
		states_provinces = feature.NaturalEarthFeature(
			category='cultural',
	#		name='admin_1_states_provinces_shp',
			name='admin_0_countries',
			scale='10m',
			facecolor='none')

		for L in range(nwk):
			wk=L+1
			#Read grads binary file size H, W  --it assumes that 2AFC file exists (template for final domain size)
			with open('../output/'+fprefix+'_'+mpref+'_2AFC_'+training_season+'_wk'+str(wki[wk-1])+'.ctl', "r") as fp:
				for line in lines_that_contain("XDEF", fp):
					W = int(line.split()[1])
					XD= float(line.split()[4])
			with open('../output/'+fprefix+'_'+mpref+'_2AFC_'+training_season+'_wk'+str(wki[wk-1])+'.ctl', "r") as fp:
				for line in lines_that_contain("YDEF", fp):
					H = int(line.split()[1])
					YD= float(line.split()[4])

			#Prepare to read grads binary file  [float32 for Fortran sequential binary files]
			Record = np.dtype(('float32', H*W))

			#B = np.fromfile('../output/'+fprefix+'_'+score+'_'+training_season+'_'+mon+str(fday)+'_wk'+str(wk)+'.dat',dtype=Record, count=-1).astype('float')
			f=open('../output/'+fprefix+'_'+score+'_'+training_season+'_'+mon+str(fday)+'_wk'+str(wki[wk-1])+'.dat','rb')

			tit=['Below Normal','Normal','Above Normal']
			for i in range(3):
					#Since CPT writes grads files in sequential format, we need to excise the 4 bytes between records (recl)
					recl=struct.unpack('i',f.read(4))[0]
					numval=int(recl/np.dtype('float32').itemsize)
					#We now read the field for that record (probabilistic files have 3 records: below, normal and above)
					B=np.fromfile(f,dtype='float32',count=numval) #astype('float')
					endrec=struct.unpack('i',f.read(4))[0]
					var = np.flip(np.transpose(B.reshape((W, H), order='F')),0)
					var[var<0]=np.nan #only positive values
					ax2=plt.subplot(nwk, 3, (L*3)+(i+1),projection=ccrs.PlateCarree())
					ax2.set_title("Week "+str(wki[wk-1])+ ": "+tit[i])
					ax2.add_feature(feature.LAND)
					ax2.add_feature(feature.COASTLINE)
					#ax2.set_ybound(lower=lati, upper=late)
					pl2=ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
						linewidth=2, color='gray', alpha=0.5, linestyle='--')
					pl2.xlabels_top = False
					pl2.ylabels_left = True
					pl2.ylabels_right = False
					pl2.xformatter = LONGITUDE_FORMATTER
					pl2.yformatter = LATITUDE_FORMATTER
					ax2.add_feature(states_provinces, edgecolor='gray')
					ax2.set_extent([loni,loni+W*XD,lati,lati+H*YD], ccrs.PlateCarree())

					#ax2.set_ybound(lower=lati, upper=late)
					#ax2.set_xbound(lower=loni, upper=lone)
					#ax2.set_adjustable('box')
					#ax2.set_aspect('auto',adjustable='datalim',anchor='C')
					CS=ax2.pcolormesh(np.linspace(loni, loni+W*XD,num=W), np.linspace(lati,lati+H*YD, num=H), var,
					vmin=0,vmax=100,
					cmap=plt.cm.bwr,
					transform=ccrs.PlateCarree())
					#plt.show(block=False)

		plt.subplots_adjust(hspace=0)
		plt.subplots_adjust(bottom=0.15, top=0.9)
		cax = plt.axes([0.2, 0.08, 0.6, 0.04])
		cbar = plt.colorbar(CS,cax=cax, orientation='horizontal')
		cbar.set_label('Probability (%)') #, rotation=270)
		f.close()

def pltmapffNC(thrs,ispctl,ntrain,loni,lone,lati,late,fprefix,mpref,training_season,mon,fday,nwk,wki):
	"""A simple function for ploting probabilistic forecasts in flexible format (for a given threshold)
	using netcdf files
	[FOR NOW, IT ONLY WORKS FOR ECMWF]

	PARAMETERS
	----------
		thrs: the threshold, in the units of the predictand
		loni: western longitude
		lone: eastern longitude
		lati: southern latitude
		late: northern latitude
	"""
	#Implement: read degrees of freedom from CPT file
	#Formally, for CCA, dof=ntrain - #CCAmodes -1 ; since ntrain is huge after concat, dof~=ntrain for now
	dof=20

	plt.figure(figsize=(15,15))

	if ispctl:
		thrso=thrs
		thrst = [x * 100 for x in thrs]

	#Create a feature for States/Admin 1 regions at 1:10m from Natural Earth
	states_provinces = feature.NaturalEarthFeature(
		category='cultural',
	#	name='admin_1_states_provinces_shp',
		name='admin_0_countries',
		scale='10m',
		facecolor='none')

	for L in range(nwk):
		wk=L+1

		#Read mu and sigma (average and std) directly from the NC files
		nc_fmu  = Dataset('../input/noMOS/modelfcst_mu_'+fprefix+'_'+mon+'_wk'+str(wki[wk-1])+'.nc', 'r')
		nc_fstd = Dataset('../input/noMOS/modelfcst_std_'+fprefix+'_'+mon+'_wk'+str(wki[wk-1])+'.nc', 'r')
		nc_omu  = Dataset('../input/noMOS/obs_mu_'+fprefix+'_'+mon+'_wk'+str(wki[wk-1])+'.nc', 'r')
		nc_ostd = Dataset('../input/noMOS/obs_std_'+fprefix+'_'+mon+'_wk'+str(wki[wk-1])+'.nc', 'r')
		nc_attrs, nc_dims, nc_vars = ncdump(nc_fmu,verb=False)
		# Extract data from NetCDF file
		lats = nc_fmu.variables['Y'][:]
		H = nc_fmu.variables['Y'].size
		YD = 1.5 #ECMWF; in the future, read it from the Y:pointwidth attribute in the NC file
		lons = nc_fmu.variables['X'][:]
		W = nc_fmu.variables['X'].size
		XD = 1.5 #ECMWF; in the future, read it from the X:pointwidth attribute in the NC file
		muf = np.squeeze(nc_fmu.variables['ratio'][:])
		vari = (np.squeeze(nc_fstd.variables['ratio'][:]))**2
		muc = np.squeeze(nc_omu.variables['tp'][:])
		varc = (np.squeeze(nc_ostd.variables['tp'][:]))**2

		#Compute scale parameter for the t-Student distribution
		scalef=np.sqrt(dof*vari)   #due to transformation from Gamma
		scalec=np.sqrt((dof-2)/dof*varc)

		if ispctl:
			thrs[wk-1]=t.ppf(thrso[wk-1], dof, loc=muc, scale=scalec)  #If using percentiles, compute value using climo

		fprob = exceedprob(thrs[wk-1],dof,muf,scalef)

		if (nwk % 2) == 0:  #is nwk even or odd?
			ax = plt.subplot(nwk/2, 2, wk, projection=ccrs.PlateCarree())
		else:
			ax = plt.subplot(1, nwk, wk, projection=ccrs.PlateCarree())  #odd nwk case
		ax.set_extent([loni,loni+W*XD,lati,lati+H*YD], ccrs.PlateCarree())

		ax.add_feature(feature.LAND)
		ax.add_feature(feature.COASTLINE)
		if ispctl:
			ax.set_title('Probability (%) of exceeding the '+str(int(thrst[wk-1]))+'th percentile for Week '+str(wki[wk-1]))
		else:
			ax.set_title('Probability (%) of exceeding '+str(thrs[wk-1])+" mm/week"+' for Week '+str(wki[wk-1]))

		pl=ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
		 	linewidth=2, color='gray', alpha=0.5, linestyle='--')
		pl.xlabels_top = False
		pl.ylabels_left = True
		pl.ylabels_right = False
		pl.xformatter = LONGITUDE_FORMATTER
		pl.yformatter = LATITUDE_FORMATTER
		ax.add_feature(states_provinces, edgecolor='gray')
		ax.set_ybound(lower=lati, upper=late)
		CS=plt.pcolormesh(np.linspace(lons[0], lons[-1],num=W), np.linspace(lats[0], lats[-1], num=H), np.squeeze(fprob),
    		vmin=0,vmax=100,
    		cmap=plt.cm.bwr,
    		transform=ccrs.PlateCarree())
		label = 'Probability (%) of Exceedance'

		plt.subplots_adjust(hspace=0)
		plt.subplots_adjust(bottom=0.15, top=0.9)
		cax = plt.axes([0.2, 0.08, 0.6, 0.04])
		cbar = plt.colorbar(CS,cax=cax, orientation='horizontal')
		cbar.set_label(label) #, rotation=270)

def pltmapff(thrs,ispctl,ntrain,loni,lone,lati,late,fprefix,mpref,training_season,mon,fday,nwk,wki):
	"""A simple function for ploting probabilistic forecasts in flexible format (for a given threshold)

	PARAMETERS
	----------
		thrs: the threshold, in the units of the predictand
		loni: western longitude
		lone: eastern longitude
		lati: southern latitude
		late: northern latitude
	"""
	if mpref=='noMOS' and fprefix=='PRCP':
		pltmapffNC(thrs,ispctl,ntrain,loni,lone,lati,late,fprefix,mpref,training_season,mon,fday,nwk,wki)
	else:
		#Implement: read degrees of freedom from CPT file
		#Formally, for CCA, dof=ntrain - #CCAmodes -1 ; since ntrain is huge after concat, dof~=ntrain for now
		dof=ntrain

		#Read grads binary file size H, W  --it assumes all files have the same size
		with open('../output/'+fprefix+'_'+mpref+'FCST_mu_'+training_season+'_'+str(mon)+str(fday)+'_wk1.ctl', "r") as fp:
			for line in lines_that_contain("XDEF", fp):
				W = int(line.split()[1])
				XD= float(line.split()[4])
		with open('../output/'+fprefix+'_'+mpref+'FCST_mu_'+training_season+'_'+str(mon)+str(fday)+'_wk1.ctl', "r") as fp:
			for line in lines_that_contain("YDEF", fp):
				H = int(line.split()[1])
				YD= float(line.split()[4])
		with open('../output/'+fprefix+'_'+mpref+'FCST_Obs_'+training_season+'_'+str(mon)+str(fday)+'_wk1.ctl', "r") as fp:
			for line in lines_that_contain("TDEF", fp):
				T = int(line.split()[1])
				TD= 1  #not used

		plt.figure(figsize=(15,15))

		if ispctl:
			thrso=thrs
			thrst = [x * 100 for x in thrs]

		for L in range(nwk):
			wk=L+1
			#Read mean
			#Since CPT writes grads files in sequential format, we need to excise the 4 bytes between records (recl)
			f=open('../output/'+fprefix+'_'+mpref+'FCST_mu_'+training_season+'_'+str(mon)+str(fday)+'_wk'+str(wki[wk-1])+'.dat','rb')
			recl=struct.unpack('i',f.read(4))[0]
			numval=int(recl/np.dtype('float32').itemsize)
			#Now we read the field
			A=np.fromfile(f,dtype='float32',count=numval)
			muf = np.transpose(A.reshape((W, H), order='F'))
			muf[muf==-999.]=np.nan #only sensible values
			# if fprefix=='RFREQ':
			# 	muf=muf/100

			#Read variance
			f=open('../output/'+fprefix+'_'+mpref+'FCST_var_'+training_season+'_'+str(mon)+str(fday)+'_wk'+str(wki[wk-1])+'.dat','rb')
			recl=struct.unpack('i',f.read(4))[0]
			numval=int(recl/np.dtype('float32').itemsize)
			#Now we read the field
			A=np.fromfile(f,dtype='float32',count=numval)
			vari = np.transpose(A.reshape((W, H), order='F'))
			vari[vari==-999.]=np.nan #only sensible values
			# if fprefix=='RFREQ':
			# 	vari=vari/100

			#Obs file--------
			#Compute obs mean and variance.
			#
			muc0=np.empty([T,H,W])  #define array for later use
			#Since CPT writes grads files in sequential format, we need to excise the 4 bytes between records (recl)
			f=open('../output/'+fprefix+'_'+mpref+'FCST_Obs_'+training_season+'_'+str(mon)+str(fday)+'_wk'+str(wki[wk-1])+'.dat','rb')
			#cycle for all time steps  (same approach to read GrADS files as before, but now read T times)
			for it in range(T):
				#Now we read the field
				recl=struct.unpack('i',f.read(4))[0]
				numval=int(recl/np.dtype('float32').itemsize) #this if for each time stamp
				A0=np.fromfile(f,dtype='float32',count=numval)
				endrec=struct.unpack('i',f.read(4))[0]  #needed as Fortran sequential repeats the header at the end of the record!!!
				muc0[it,:,:]= np.transpose(A0.reshape((W, H), order='F'))

			muc0[muc0==-999.]=np.nan #identify NaNs
			muc=np.nanmean(muc0, axis=0)  #axis 0 is T
			#Compute obs variance
			varc=np.nanvar(muc0, axis=0)  #axis 0 is T

			#Compute scale parameter for the t-Student distribution
			scalef=np.sqrt(dof*vari)   #due to transformation from Gamma
			scalec=np.sqrt((dof-2)/dof*varc)

			if ispctl:
				thrs[wk-1]=t.ppf(thrso[wk-1], dof, loc=muc, scale=scalec)  #If using percentiles, compute value using climo

			fprob = exceedprob(thrs[wk-1],dof,muf,scalef)

			if (nwk % 2) == 0:  #is nwk even or odd?
				ax = plt.subplot(nwk/2, 2, wk, projection=ccrs.PlateCarree())
			else:
				ax = plt.subplot(nwk, 1, wk, projection=ccrs.PlateCarree())  #odd nwk case

			ax.set_extent([loni,loni+W*XD,lati,lati+H*YD], ccrs.PlateCarree())

			#Create a feature for States/Admin 1 regions at 1:10m from Natural Earth
			states_provinces = feature.NaturalEarthFeature(
				category='cultural',
	#			name='admin_1_states_provinces_shp',
				name='admin_0_countries',
				scale='10m',
				facecolor='none')

			ax.add_feature(feature.LAND)
			ax.add_feature(feature.COASTLINE)
			if ispctl:
				ax.set_title('Probability (%) of exceeding the '+str(int(thrst[wk-1]))+'th percentile for Week '+str(wki[wk-1]))
			else:
				ax.set_title('Probability (%) of exceeding '+str(thrs[wk-1])+" mm/week"+' for Week '+str(wki[wk-1]))

			pl=ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
			 	linewidth=2, color='gray', alpha=0.5, linestyle='--')
			pl.xlabels_top = False
			pl.ylabels_left = True
			pl.ylabels_right = False
			pl.xformatter = LONGITUDE_FORMATTER
			pl.yformatter = LATITUDE_FORMATTER
			ax.add_feature(states_provinces, edgecolor='gray')
			ax.set_ybound(lower=lati, upper=late)
			CS=plt.pcolormesh(np.linspace(loni, loni+W*XD,num=W), np.linspace(lati+H*YD, lati, num=H), fprob,
	    		vmin=0,vmax=100,
	    		cmap=plt.cm.bwr,
	    		transform=ccrs.PlateCarree())
			label = 'Probability (%) of Exceedance'

			plt.subplots_adjust(hspace=0)
			plt.subplots_adjust(bottom=0.15, top=0.9)
			cax = plt.axes([0.2, 0.08, 0.6, 0.04])
			cbar = plt.colorbar(CS,cax=cax, orientation='horizontal')
			cbar.set_label(label) #, rotation=270)
			f.close()

def pltprobffNC(thrsn,ispctl,ntrain,lon,lat,loni,lone,lati,late,fprefix,mpref,training_season,mon,fday,nwk,wki):
	"""A simple function for ploting probabilities of exceedance and PDFs (for a given threshold)

	PARAMETERS
	----------
		thrs: the threshold, in the units of the predictand
		lon: longitude
		lat: latitude
	"""
	#Implement: read degrees of freedom from CPT file
	#Formally, for CCA, dof=ntrain - #CCAmodes -1 ; since ntrain is huge after concat, dof~=ntrain for now
	dof=20
	thrs=thrsn

	nc_fmu  = Dataset('../input/noMOS/modelfcst_mu_'+fprefix+'_'+mon+'_wk1.nc', 'r')
	#nc_attrs, nc_dims, nc_vars = ncdump(nc_fmu,verb=False)
	# Extract data from NetCDF file
	lats = nc_fmu.variables['Y'][:]
	H = nc_fmu.variables['Y'].size
	YD = 1.5 #ECMWF; in the future, read it from the Y:pointwidth attribute in the NC file
	lons = nc_fmu.variables['X'][:]
	W = nc_fmu.variables['X'].size
	XD = 1.5 #ECMWF; in the future, read it from the X:pointwidth attribute in the NC file

	#Find the gridbox:
	lonrange = np.linspace(lons[0], lons[-1],num=W)
	latrange = np.linspace(lats[0], lats[-1], num=H)  #need to reverse the latitudes because of CPT (GrADS YREV option)
	lon_grid, lat_grid = np.meshgrid(lonrange, latrange)
	a = abs(lat_grid-lat)+abs(lon_grid-lon)
	i,j = np.unravel_index(a.argmin(),a.shape)   #i:latitude   j:longitude

	#Now compute stuff and plot
	plt.figure(figsize=(15,15))

	thrso=thrs

	#Fix x-axis to that of week 4, expected to have higher spread
		#Read mu and sigma (average and std) directly from the NC files
	nc_fmu  = Dataset('../input/noMOS/modelfcst_mu_'+fprefix+'_'+mon+'_wk'+str(wki[3])+'.nc', 'r')
	nc_fstd = Dataset('../input/noMOS/modelfcst_std_'+fprefix+'_'+mon+'_wk'+str(wki[3])+'.nc', 'r')
	nc_omu  = Dataset('../input/noMOS/obs_mu_'+fprefix+'_'+mon+'_wk'+str(wki[3])+'.nc', 'r')
	nc_ostd = Dataset('../input/noMOS/obs_std_'+fprefix+'_'+mon+'_wk'+str(wki[3])+'.nc', 'r')
	nc_attrs, nc_dims, nc_vars = ncdump(nc_fmu,verb=False)
		# Extract data from NetCDF file
	muf4 = np.squeeze(nc_fmu.variables['ratio'][:])
	muf4 = muf4[i,j]
	varf4 = (np.squeeze(nc_fstd.variables['ratio'][:]))**2
	varf4 = varf4[i,j]
	muc4 = np.squeeze(nc_omu.variables['tp'][:])
	muc4 = muc4[i,j]
	varc4 = (np.squeeze(nc_ostd.variables['tp'][:]))**2
	varc4 = varc4[i,j]
		#Compute scale parameter for the t-Student distribution
	scalef4=np.sqrt(dof*varf4)   #due to transformation from Gamma
	scalec4=np.sqrt((dof-2)/dof*varc4)
	x = np.linspace(min(t.ppf(0.00001, dof, loc=muf4, scale=scalef4),t.ppf(0.00001, dof, loc=muc4, scale=scalec4)),max(t.ppf(0.9999, dof, loc=muf4, scale=scalef4),t.ppf(0.9999, dof, loc=muc4, scale=scalec4)), 100)


	for L in range(nwk):
		wk=L+1
		#Read mu and sigma (average and std) directly from the NC files
		nc_fmu  = Dataset('../input/noMOS/modelfcst_mu_'+fprefix+'_'+mon+'_wk'+str(wki[wk-1])+'.nc', 'r')
		nc_fstd = Dataset('../input/noMOS/modelfcst_std_'+fprefix+'_'+mon+'_wk'+str(wki[wk-1])+'.nc', 'r')
		nc_omu  = Dataset('../input/noMOS/obs_mu_'+fprefix+'_'+mon+'_wk'+str(wki[wk-1])+'.nc', 'r')
		nc_ostd = Dataset('../input/noMOS/obs_std_'+fprefix+'_'+mon+'_wk'+str(wki[wk-1])+'.nc', 'r')
		nc_attrs, nc_dims, nc_vars = ncdump(nc_fmu,verb=False)
		# Extract data from NetCDF file
		muf = np.squeeze(nc_fmu.variables['ratio'][:])
		muf=muf[i,j]
		varf = (np.squeeze(nc_fstd.variables['ratio'][:]))**2
		varf=varf[i,j]
		muc = np.squeeze(nc_omu.variables['tp'][:])
		muc=muc[i,j]
		varc = (np.squeeze(nc_ostd.variables['tp'][:]))**2
		varc=varc[i,j]

		#Compute scale parameter for the t-Student distribution
		scalef=np.sqrt(dof*varf)   #due to transformation from Gamma
		scalec=np.sqrt((dof-2)/dof*varc)

		if ispctl:
			thrs[wk-1]=t.ppf(thrso[wk-1], dof, loc=muc, scale=scalec)  #If using percentiles, compute value using climo
			#print('Week '+str(wk)+': percentile '+str(int(thrso[wk-1]))+' is '+str(np.round(thrs[wk-1]))+' mm')

		#Original case: dynamic x-axis
		#x = np.linspace(min(t.ppf(0.00001, dof, loc=muf, scale=scalef),t.ppf(0.00001, dof, loc=muc, scale=scalec)),max(t.ppf(0.9999, dof, loc=muf, scale=scalef),t.ppf(0.9999, dof, loc=muc, scale=scalec)), 100)

		style = dict(size=10, color='black')

		#cprob = special.erfc((x-muc)/scalec)
		cprob = exceedprob(thrs[wk-1],dof,muc,scalec)
		fprob = exceedprob(thrs[wk-1],dof,muf,scalef)
		cprobth = np.round(t.sf(thrs[wk-1], dof, loc=muc, scale=scalec)*100,2)
		fprobth = np.round(t.sf(thrs[wk-1], dof, loc=muf, scale=scalef)*100,2)
		cpdf=t.pdf(x, dof, loc=muc, scale=scalec)*100
		fpdf=t.pdf(x, dof, loc=muf, scale=scalef)*100
		oddsrc =(fprobth/cprobth)

		fig, ax = plt.subplots(1, 2,figsize=(12,4))
		#font = {'family' : 'Palatino',
		#        'size'   : 16}
		#plt.rc('font', **font)
		#plt.rc('text', usetex=True)
		#plt.rc('font', family='serif')

		plt.subplot(1, 2, 1)
		plt.plot(x, t.sf(x, dof, loc=muc, scale=scalec)*100,'b-', lw=5, alpha=0.6, label='clim')
		plt.plot(x, t.sf(x, dof, loc=muf, scale=scalef)*100,'r-', lw=5, alpha=0.6, label='fcst')
		plt.axvline(x=thrs[wk-1], color='k', linestyle='--')
		plt.plot(thrs[wk-1], fprobth,'ok')
		plt.plot(thrs[wk-1], cprobth,'ok')
		plt.text(thrs[wk-1]+0.05, cprobth, str(cprobth)+'%', **style)
		plt.text(thrs[wk-1]+0.05, fprobth, str(fprobth)+'%', **style)
		#plt.text(0.1, 10, r'$\frac{P(fcst)}{P(clim)}=$'+str(round(oddsrc,1)), **style)
		plt.text(min(t.ppf(0.0001, dof, loc=muf4, scale=scalef4),t.ppf(0.0001, dof, loc=muc4, scale=scalec4)), -20, 'P(fcst)/P(clim)='+str(round(oddsrc,1)), **style)
		plt.legend(loc='best', frameon=False)
		# Add title and axis names
		plt.title('Probabilities of Exceedance for Week '+str(wk))
		plt.xlabel('Rainfall')
		plt.ylabel('Probability (%)')
		# Limits for the X axis
		plt.xlim(min(t.ppf(0.00001, dof, loc=muf4, scale=scalef4),t.ppf(0.00001, dof, loc=muc4, scale=scalec4)),max(t.ppf(0.9999, dof, loc=muf4, scale=scalef4),t.ppf(0.9999, dof, loc=muc4, scale=scalec4)))

		plt.subplot(1, 2, 2)
		plt.plot(x, cpdf,'b-', lw=5, alpha=0.6, label='clim')
		plt.plot(x, fpdf,'r-', lw=5, alpha=0.6, label='fcst')
		plt.axvline(x=thrs[wk-1], color='k', linestyle='--')
		#fill area under the curve --not done
		#section = np.arange(min(t.ppf(0.00001, dof, loc=muf, scale=scalef),t.ppf(0.00001, dof, loc=muc, scale=scalec)), thrs, 1/20.)
		#plt.fill_between(section,f(section))
		plt.legend(loc='best', frameon=False)
		# Add title and axis names
		plt.title('Probability Density Functions for Week '+str(wk))
		plt.xlabel('Rainfall')
		plt.ylabel('Density')
		# Limits for the X axis
		plt.xlim(min(t.ppf(0.00001, dof, loc=muf4, scale=scalef4),t.ppf(0.00001, dof, loc=muc4, scale=scalec4)),max(t.ppf(0.9999, dof, loc=muf4, scale=scalef4),t.ppf(0.9999, dof, loc=muc4, scale=scalec4)))

	plt.subplots_adjust(hspace=0)
	plt.subplots_adjust(bottom=0.15, top=0.9)
	#cax = plt.axes([0.2, 0.08, 0.6, 0.04])
	#cbar = plt.colorbar(CS,cax=cax, orientation='horizontal')
	#cbar.set_label(label) #, rotation=270)

def pltprobff(thrsn,ispctl,ntrain,lon,lat,loni,lone,lati,late,fprefix,mpref,training_season,mon,fday,nwk,wki):
	"""A simple function for ploting probabilities of exceedance and PDFs (for a given threshold)

	PARAMETERS
	----------
		thrs: the threshold, in the units of the predictand
		lon: longitude
		lat: latitude
	"""
	if mpref=='noMOS' and fprefix=='PRCP':
		pltprobffNC(thrsn,ispctl,ntrain,lon,lat,loni,lone,lati,late,fprefix,mpref,training_season,mon,fday,nwk,wki)
	else:
		#Implement: read degrees of freedom from CPT file
		#Formally, for CCA, dof=ntrain - #CCAmodes -1 ; since ntrain is huge after concat, dof~=ntrain for now
		dof=ntrain
		thrs=thrsn


		#Read grads binary file size H, W  --it assumes all files have the same size, and that 2AFC exists
		with open('../output/'+fprefix+'_'+mpref+'_2AFC_'+training_season+'_wk1.ctl', "r") as fp:
			for line in lines_that_contain("XDEF", fp):
				W = int(line.split()[1])
				XD= float(line.split()[4])
		with open('../output/'+fprefix+'_'+mpref+'_2AFC_'+training_season+'_wk1.ctl', "r") as fp:
			for line in lines_that_contain("YDEF", fp):
				H = int(line.split()[1])
				YD= float(line.split()[4])
		with open('../output/'+fprefix+'_'+mpref+'FCST_Obs_'+training_season+'_'+str(mon)+str(fday)+'_wk1.ctl', "r") as fp:
			for line in lines_that_contain("TDEF", fp):
				T = int(line.split()[1])
				TD= 1  #not used

		#Find the gridbox:
		lonrange = np.linspace(loni, loni+W*XD,num=W)
		latrange = np.linspace(lati+H*YD, lati, num=H)  #need to reverse the latitudes because of CPT (GrADS YREV option)
		lon_grid, lat_grid = np.meshgrid(lonrange, latrange)
		a = abs(lat_grid-lat)+abs(lon_grid-lon)
		i,j = np.unravel_index(a.argmin(),a.shape)   #i:latitude   j:longitude

		#Now compute stuff and plot
		plt.figure(figsize=(15,15))

		thrso=thrs

		for L in range(nwk):
			wk=L+1
			#Forecast files--------
			#Read mean
			#Since CPT writes grads files in sequential format, we need to excise the 4 bytes between records (recl)
			f=open('../output/'+fprefix+'_'+mpref+'FCST_mu_'+training_season+'_'+str(mon)+str(fday)+'_wk'+str(wki[wk-1])+'.dat','rb')
			recl=struct.unpack('i',f.read(4))[0]
			numval=int(recl/np.dtype('float32').itemsize)
			#Now we read the field
			A=np.fromfile(f,dtype='float32',count=numval)
			muf = np.transpose(A.reshape((W, H), order='F'))
			muf[muf==-999.]=np.nan #identify NaNs
			muf=muf[i,j]
			if fprefix=='RFREQ':
				muf=muf/100

			#Read variance
			f=open('../output/'+fprefix+'_'+mpref+'FCST_var_'+training_season+'_'+str(mon)+str(fday)+'_wk'+str(wki[wk-1])+'.dat','rb')
			recl=struct.unpack('i',f.read(4))[0]
			numval=int(recl/np.dtype('float32').itemsize)
			#Now we read the field
			A=np.fromfile(f,dtype='float32',count=numval)
			varf = np.transpose(A.reshape((W, H), order='F'))
			varf[varf==-999.]=np.nan #identify NaNs
			varf=varf[i,j]
			if fprefix=='RFREQ':
				varf=varf/10000

			#Obs file--------
			#Compute obs mean and variance.
			#
			muc0=np.empty([T,H,W])  #define array for later use
			#Since CPT writes grads files in sequential format, we need to excise the 4 bytes between records (recl)
			f=open('../output/'+fprefix+'_'+mpref+'FCST_Obs_'+training_season+'_'+str(mon)+str(fday)+'_wk'+str(wki[wk-1])+'.dat','rb')
			#cycle for all time steps  (same approach to read GrADS files as before, but now read T times)
			for it in range(T):
				#Now we read the field
				recl=struct.unpack('i',f.read(4))[0]
				numval=int(recl/np.dtype('float32').itemsize) #this if for each time stamp
				A0=np.fromfile(f,dtype='float32',count=numval)
				endrec=struct.unpack('i',f.read(4))[0]  #needed as Fortran sequential repeats the header at the end of the record!!!
				muc0[it,:,:]= np.transpose(A0.reshape((W, H), order='F'))

			muc0[muc0==-999.]=np.nan #identify NaNs
			muc=np.nanmean(muc0, axis=0)  #axis 0 is T
			#Compute obs variance
			varc=np.nanvar(muc0, axis=0)  #axis 0 is T
			#Select gridbox values
			muc=muc[i,j]
			#print(muc)   #Test it's actually zero
			varc=varc[i,j]

			#Compute scale parameter for the t-Student distribution
			scalef=np.sqrt(dof*varf)   #due to transformation from Gamma
			scalec=np.sqrt((dof-2)/dof*varc)

			if ispctl:
				thrs[wk-1]=t.ppf(thrso[wk-1], dof, loc=muc, scale=scalec)  #If using percentiles, compute value using climo
				#print('Week '+str(wk)+': percentile '+str(int(thrso[wk-1]))+' is '+str(np.round(thrs[wk-1]))+' mm')

			x = np.linspace(min(t.ppf(0.00001, dof, loc=muf, scale=scalef),t.ppf(0.00001, dof, loc=muc, scale=scalec)),max(t.ppf(0.9999, dof, loc=muf, scale=scalef),t.ppf(0.9999, dof, loc=muc, scale=scalec)), 100)

			style = dict(size=10, color='black')

			#cprob = special.erfc((x-muc)/scalec)
			cprob = exceedprob(thrs[wk-1],dof,muc,scalec)
			fprob = exceedprob(thrs[wk-1],dof,muf,scalef)
			cprobth = np.round(t.sf(thrs[wk-1], dof, loc=muc, scale=scalec)*100,2)
			fprobth = np.round(t.sf(thrs[wk-1], dof, loc=muf, scale=scalef)*100,2)
			cpdf=t.pdf(x, dof, loc=muc, scale=scalec)*100
			fpdf=t.pdf(x, dof, loc=muf, scale=scalef)*100
			oddsrc =(fprobth/cprobth)

			fig, ax = plt.subplots(1, 2,figsize=(12,4))
			#font = {'family' : 'Palatino',
			#        'size'   : 16}
			#plt.rc('font', **font)
			#plt.rc('text', usetex=True)
			#plt.rc('font', family='serif')

			plt.subplot(1, 2, 1)
			plt.plot(x, t.sf(x, dof, loc=muc, scale=scalec)*100,'b-', lw=5, alpha=0.6, label='clim')
			plt.plot(x, t.sf(x, dof, loc=muf, scale=scalef)*100,'r-', lw=5, alpha=0.6, label='fcst')
			plt.axvline(x=thrs[wk-1], color='k', linestyle='--')
			plt.plot(thrs[wk-1], fprobth,'ok')
			plt.plot(thrs[wk-1], cprobth,'ok')
			plt.text(thrs[wk-1]+0.05, cprobth, str(cprobth)+'%', **style)
			plt.text(thrs[wk-1]+0.05, fprobth, str(fprobth)+'%', **style)
			#plt.text(0.1, 10, r'$\frac{P(fcst)}{P(clim)}=$'+str(round(oddsrc,1)), **style)
			plt.text(min(t.ppf(0.0001, dof, loc=muf, scale=scalef),t.ppf(0.0001, dof, loc=muc, scale=scalec)), -20, 'P(fcst)/P(clim)='+str(round(oddsrc,1)), **style)
			plt.legend(loc='best', frameon=False)
			# Add title and axis names
			plt.title('Probabilities of Exceedance for Week '+str(wki[wk-1]))
			if fprefix=='PRCP':
				plt.xlabel('Rainfall')
			elif fprefix=='RFREQ':
				plt.xlabel('Rainfall freq.')
			plt.ylabel('Probability (%)')
			# Limits for the X axis
			plt.xlim(min(t.ppf(0.00001, dof, loc=muf, scale=scalef),t.ppf(0.00001, dof, loc=muc, scale=scalec)),max(t.ppf(0.9999, dof, loc=muf, scale=scalef),t.ppf(0.9999, dof, loc=muc, scale=scalec)))

			plt.subplot(1, 2, 2)
			plt.plot(x, cpdf,'b-', lw=5, alpha=0.6, label='clim')
			plt.plot(x, fpdf,'r-', lw=5, alpha=0.6, label='fcst')
			plt.axvline(x=thrs[wk-1], color='k', linestyle='--')
			#fill area under the curve --not done
			#section = np.arange(min(t.ppf(0.00001, dof, loc=muf, scale=scalef),t.ppf(0.00001, dof, loc=muc, scale=scalec)), thrs, 1/20.)
			#plt.fill_between(section,f(section))
			plt.legend(loc='best', frameon=False)
			# Add title and axis names
			plt.title('Probability Density Functions for Week '+str(wk))
			if fprefix=='PRCP':
				plt.xlabel('Rainfall')
			elif fprefix=='RFREQ':
				plt.xlabel('Rainfall freq.')
			plt.ylabel('Density')
			# Limits for the X axis
			plt.xlim(min(t.ppf(0.00001, dof, loc=muf, scale=scalef),t.ppf(0.00001, dof, loc=muc, scale=scalec)),max(t.ppf(0.9999, dof, loc=muf, scale=scalef),t.ppf(0.9999, dof, loc=muc, scale=scalec)))

		plt.subplots_adjust(hspace=0)
		plt.subplots_adjust(bottom=0.15, top=0.9)
		#cax = plt.axes([0.2, 0.08, 0.6, 0.04])
		#cbar = plt.colorbar(CS,cax=cax, orientation='horizontal')
		#cbar.set_label(label) #, rotation=270)
		f.close()

def readNetCDF_predictand(infile,outfile, predictand, wlo2, elo2, sla2, nla2, tar):
	"""Function to read the user's predictand NetCDF file and write to CPT format.

	PARAMETERS
	----------
	predictand: a DataArray with dimensions T,Y,X
	"""

	ds=xr.open_dataset(infile,decode_times=False)
	da=list(ds.coords)

	for i in range(len(da)):
		if da[i]=='X' or da[i]=='lon' or da[i]=='longitude':
			ds = ds.rename({da[i]:'X'})
		if da[i]=='Y' or da[i]=='lat' or da[i]=='latitude':
			ds = ds.rename({da[i]:'Y'})
		if da[i]=='T' or da[i]=='time':
			deltastyr=int(ds[da[i]][0]/12)
			ds = ds.rename({da[i]:'time'})
			nmon=ds.time.shape[0]
			nyr=int(nmon/12)
			if 'months since' in ds.time.units:
				line=ds.time.units
				stdate=str(int(line.split()[2][:4])+deltastyr)+line.split()[2][-6:]
				ds['time'] = pd.date_range(stdate, periods=ds.time.shape[0], freq='M')

#	ds1=ds.sel(X=slice(wlo2,elo2),Y=slice(sla2,nla2))
	ds1_tmp=ds.sel(X=slice(wlo2,elo2),Y=slice(sla2,nla2))
	ds1=ds1_tmp.reindex(Y=ds1_tmp.Y[::-1]) #Y from N to S
	Xarr=ds1.X.values
	Yarr=ds1.Y.values
	W=ds1.X.shape[0]
	H=ds1.Y.shape[0]
	var1=ds1[predictand]
	units=ds[predictand].units
	Ti=int(ds.time.dt.year[0])
	vari = predictand
	varname = vari
	if 'True' in np.isnan(var):
	        var[np.isnan(var)]=-999. #use CPT missing value

	monthdic = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
	mi=monthdic[tar.split("-")[0]]
	mf=monthdic[tar.split("-")[1]]

	if mi==str(11):
		var1_N=var1[(var1.time.dt.month==11)]
		var1_N1=var1_N.groupby(var1_N.time.dt.year).mean('time').sel(year=slice(Ti,Ti+nyr-2))
		var1_D=var1[(var1.time.dt.month==12)]
		var1_D1=var1_D.groupby(var1_D.time.dt.year).mean('time').sel(year=slice(Ti,Ti+nyr-2))
		var1_J=var1[(var1.time.dt.month==1)]
		var1_J1=var1_J.groupby(var1_J.time.dt.year).mean('time').sel(year=slice(Ti+1,Ti+nyr-1))
		var=np.zeros(var1_D1.shape)
		for i in range(len(var1_D1.year)):
			var[i,:,:]=(var1_N1[i,:,:]+var1_D1[i,:,:]+var1_J1[i,:,:])/3.
	elif mi==str(12):
		var1_D=var1[(var1.time.dt.month==12)]
		var1_D1=var1_D.groupby(var1_D.time.dt.year).mean('time').sel(year=slice(Ti,Ti+nyr-2))
		var1_J=var1[(var1.time.dt.month==1)]
		var1_J1=var1_J.groupby(var1_J.time.dt.year).mean('time').sel(year=slice(Ti+1,Ti+nyr-1))
		var1_F = var1[(var1.time.dt.month==2)]
		var1_F1=var1_F.groupby(var1_F.time.dt.year).mean('time').sel(year=slice(Ti+1,Ti+nyr-1))
		var=np.zeros(var1_D1.shape)
		for i in range(len(var1_D1.year)):
			var[i,:,:]=(var1_D1[i,:,:]+var1_J1[i,:,:]+var1_F1[i,:,:])/3.
	else:
		var1_season = var1[(var1.time.dt.month>=mi)&(var1.time.dt.month<=mf)]
		var=var1_season.groupby(var1_season.time.dt.year).mean(dim=('time')).sel(year=slice(Ti+1,Ti+nyr-1))
	if tar=='Dec-Feb' or tar=='Nov-Jan':  #double check years are sync
		Ti=Ti
		xyear=True  #flag a cross-year season
	else:
		Ti=Ti+1
		xyear=False

	T=nyr-1
	Tarr = np.arange(Ti, Ti+T)

        #Now write the CPT file
	outfile="usr_"+predictand+"_"+tar+".tsv"
	f = open(outfile, 'w')
	f.write("xmlns:cpt=http://iri.columbia.edu/CPT/v10/\n")
	f.write("cpt:nfields=1\n")
	for it in range(T):
		if xyear==True:
			f.write("cpt:field="+vari+", cpt:T="+str(Tarr[it])+"-"+mi+"/"+str(Tarr[it]+1)+"-"+mf+", cpt:nrow="+str(H)+", cpt:ncol="+str(W)+", cpt:row=Y, cpt:col=X, cpt:units="+units+", cpt:missing=-999.\n")
		else:
			f.write("cpt:field="+vari+", cpt:T="+str(Tarr[it])+"-"+mi+"/"+mf+", cpt:nrow="+str(H)+", cpt:ncol="+str(W)+", cpt:row=Y, cpt:col=X, cpt:units="+units+", cpt:missing=-999.\n")
		np.savetxt(f, Xarr[0:-1], fmt="%.3f",newline='\t')
		f.write("\n") #next line
		for iy in range(H):
			np.savetxt(f,np.r_[Yarr[iy],var[it,iy,0:]],fmt="%.3f", newline='\t')  #excise extra line
			f.write("\n") #next line
	f.close()

def readNetCDF_Hindcasts(infile, outfile, wlo1, elo1, sla1, nla1, tgti, tgtf, mon, tar):
	"""Function to read the user's Hindcasts NetCDF file and write to CPT format.

	PARAMETERS
	----------
		Hindcats: a DataArray with dimensions S,M,L,Y,X
	"""
	ds=xr.open_dataset(infile,decode_times=False)
	da=list(ds.coords)

	for i in range(len(da)):
		if da[i]=='X' or da[i]=='lon' or da[i]=='longitude':
			ds = ds.rename({da[i]:'X'})
		if da[i]=='Y' or da[i]=='lat' or da[i]=='latitude':
			ds = ds.rename({da[i]:'Y'})
		if da[i]=='S':
			deltastyr=int(ds[da[i]][0]/12)
			nmon=ds.S.shape[0]
			nyr=int(nmon/12)
		if 'months since' in ds.S.units:
				line=ds.S.units
				stdate=str(int(line.split()[2][:4])+deltastyr)+line.split()[2][-6:]
				ds['S'] = pd.date_range(stdate, periods=ds.S.shape[0], freq='M')

	ds1=ds.sel(X=slice(wlo1,elo1),Y=slice(sla1,nla1),L=slice(float(tgti),float(tgtf))).mean(dim='L',skipna=True)
	ds2=ds1.mean(dim='M',skipna=True)
	Xarr=ds2.X.values
	Yarr=ds2.Y.values
	W=ds2.X.shape[0]
	H=ds2.Y.shape[0]
	a=list(ds)

	var1=ds2[a[0]]
	units=ds[a[0]].units
	Ti=1982

	vari = a[0]
	varname = vari
	L=0.5*(float(tgtf)+float(tgti))

	monthdic = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
	S1=monthdic[mon]
	mi=monthdic[tar.split("-")[0]]
	mf=monthdic[tar.split("-")[1]]

	var1_stmon=var1[(var1.S.dt.month==int(monthdic[mon]))]
	var=var1_stmon.groupby(var1_stmon.S.dt.year).mean(dim=('S')).sel(year=slice(1982,2009))
	var_N2S=var.reindex(Y=var.Y[::-1]) #Y from N to S
	Yarr=var_N2S.Y.values
	if tar=='Dec-Feb' or tar=='Nov-Jan':  #double check years are sync
		xyear=True  #flag a cross-year season
	else:
		xyear=False
	T=2009-1982+1
	Tarr = np.arange(Ti, Ti+T)

	if 'True' in np.isnan(var):
		var[np.isnan(var)]=-999. #use CPT missing value
        #Now write the CPT file
	outfile="usr_"+a[0]+"_"+tar+"_ini"+mon+".tsv"
	f = open(outfile, 'w')
	f.write("xmlns:cpt=http://iri.columbia.edu/CPT/v10/\n")
	f.write("cpt:nfields=1\n")

	for it in range(T):
		if xyear==True:
			f.write("cpt:field="+vari+", cpt:L="+str(L)+" months, cpt:S="+str(Tarr[it])+"-"+S1+"-01T00:00, cpt:T="+str(Tarr[it])+"-"+mi+"/"+str(Tarr[it]+1)+"-"+mf+", cpt:nrow="+str(H)+", cpt:ncol="+str(W)+", cpt:row=Y, cpt:col=X, cpt:units="+units+", cpt:missing=-999.\n")
		else:
			f.write("cpt:field="+vari+", cpt:L="+str(L)+" months, cpt:S="+str(Tarr[it])+"-"+S1+"-01T00:00, cpt:T="+str(Tarr[it])+"-"+mi+"/"+mf+", cpt:nrow="+str(H)+", cpt:ncol="+str(W)+", cpt:row=Y, cpt:col=X, cpt:units="+units+", cpt:missing=-999.\n")
	np.savetxt(f, Xarr, fmt="%.3f",newline='\t')
	f.write("\n") #next line
	for iy in range(H):
		np.savetxt(f,np.r_[Yarr[iy],var_N2S[it,iy,0:]],fmt="%.3f", newline='\t')  #excise extra line
		f.write("\n") #next line
	f.close()

def readNetCDF_Forecast(infile, outfile, monf, fyr, tgti, tgtf, tar, wlo1, elo1, sla1, nla1):
        """Function to read the user's forecast NetCDF file and write to CPT format.

        PARAMETERS
        ----------
                Forecat: a DataArray with dimensions S,M,L,Y,X
        """
        ds=xr.open_dataset(infile,decode_times=False)
        da=list(ds.coords)

        for i in range(len(da)):
                if da[i]=='X' or da[i]=='lon' or da[i]=='longitude':
                        ds = ds.rename({da[i]:'X'})
                if da[i]=='Y' or da[i]=='lat' or da[i]=='latitude':
                        ds = ds.rename({da[i]:'Y'})
                if da[i]=='S':
                        deltastyr=int(ds[da[i]][0]/12)
                        nmon=ds.S.shape[0]
                        nyr=int(nmon/12)
                        if 'months since' in ds.S.units:
                                line=ds.S.units
                                stdate=str(int(line.split()[2][:4])+deltastyr)+line.split()[2][-6:]
                                ds['S'] = pd.date_range(stdate, periods=ds.S.shape[0], freq='M')

        ds1=ds.sel(X=slice(wlo1,elo1),Y=slice(sla1,nla1),L=slice(float(tgti),float(tgtf))).mean(dim='L',skipna=True)
        ds2=ds1.mean(dim='M',skipna=True)
        Xarr=ds2.X.values
        Yarr=ds2.Y.values
        W=ds2.X.shape[0]
        H=ds2.Y.shape[0]
        a=list(ds)

        var1=ds2[a[0]]
        units=ds[a[0]].units
        Ti=fyr

        vari = a[0]
        varname = vari
        L=0.5*(float(tgtf)+float(tgti))

        monthdic = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
        S1=monthdic[monf]
        mi=monthdic[tar.split("-")[0]]
        mf=monthdic[tar.split("-")[1]]

        var1_stmon=var1[(var1.S.dt.month==int(monthdic[monf]))]
        var=var1_stmon.groupby(var1_stmon.S.dt.year).mean(dim=('S')).sel(year=fyr)
        var_N2S=var.reindex(Y=var.Y[::-1])
        Yarr=var_N2S.Y.values
        if tar=='Dec-Feb' or tar=='Nov-Jan':  #double check years are sync
                xyear=True  #flag a cross-year season
        else:
                xyear=False
        T=1
        Tarr = np.arange(Ti, Ti+T)

        if 'True' in np.isnan(var):
                var[np.isnan(var)]=-999. #use CPT missing value
        #Now write the CPT file
        outfile="usr_fcst_"+a[0]+"_"+tar+"_ini"+monf+str(fyr)+".tsv"
        f = open(outfile, 'w')
        f.write("xmlns:cpt=http://iri.columbia.edu/CPT/v10/\n")
        f.write("cpt:nfields=1\n")

        for it in range(T):
                if xyear==True:
                        f.write("cpt:field="+vari+", cpt:L="+str(L)+" months, cpt:S="+str(Tarr[it])+"-"+S1+"-01T00:00, cpt:T="+str(Tarr[it])+"-"+mi+"/"+str(Tarr[it]+1)+"-"+mf+", cpt:nrow="+str(H)+", cpt:ncol="+str(W)+", cpt:row=Y, cpt:col=X, cpt:units="+units+", cpt:missing=-999.\n")
                else:
                        f.write("cpt:field="+vari+", cpt:L="+str(L)+" months, cpt:S="+str(Tarr[it])+"-"+S1+"-01T00:00, cpt:T="+str(Tarr[it])+"-"+mi+"/"+mf+", cpt:nrow="+str(H)+", cpt:ncol="+str(W)+", cpt:row=Y, cpt:col=X, cpt:units="+units+", cpt:missing=-999.\n")
        np.savetxt(f, Xarr, fmt="%.3f",newline='\t')
        f.write("\n") #next line
        for iy in range(H):
                np.savetxt(f,np.r_[Yarr[iy],var_N2S[iy,0:]],fmt="%.3f", newline='\t')  #excise extra line
                f.write("\n") #next line
        f.close()

def GetHindcasts(wlo1, elo1, sla1, nla1, day1, day2, fyr, mon, os, key, week, nlag, nday, training_season, hstep, model, hdate_last, force_download):
	nwi=4  #number of weeks to use for real-time ECMWF training period (2 initializations per week) --Send to namelist in the future
	if not force_download:
		try:
			ff=open("model_precip_"+mon+"_wk"+str(week)+".tsv", 'r')
			s = ff.readline()
		except OSError as err:
			print("\033[1mWarning:\033[0;0m {0}".format(err))
			print("Hindcasts file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
			force_download = True
	if force_download:
		#dictionary:
#		dic = { 'CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag/M%5Daverage/3./mul/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/add/4./div/S/('+training_season+')/VALUES/S/'+str(hstep)+'/STEP/dup/S/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.S/replaceGRID/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/c://name//water_density/def/998/%28kg/m3%29/:c/div//mm/unitconvert//name/(tp)/def/grid://name/%28T%29/def//units/%28months%20since%201960-01-01%29/def//standard_name/%28time%29/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/:grid/use_as_grid//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
		dic = { 'CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag/M%5Daverage/3./mul/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/add/4./div/S/('+mon+')/VALUES/S/'+str(hstep)+'/STEP/dup/S/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.S/replaceGRID/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/c://name//water_density/def/998/%28kg/m3%29/:c/div//mm/unitconvert//name/(tp)/def/grid://name/%28T%29/def//units/%28months%20since%201960-01-01%29/def//standard_name/%28time%29/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/:grid/use_as_grid//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
				'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%201%20'+mon+'%20'+str(fyr)+')%20(2300%2028%20'+mon+'%20'+str(fyr)+')/RANGE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(fyr-1)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/T/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2060/ensotime/%3Agrid/replaceGRID//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
				'ECMWFrt': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(last%206%20'+str(nwi)+'%20mul%20sub)%20(last)/RANGE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(fyr-1)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/T/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2060/ensotime/%3Agrid/replaceGRID//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
				'GEFS':
'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.EMC/.GEFS/.hindcast/.pr/S/(0000%206%20Jan%201999)/(0000%2028%20Dec%202015)/RANGEEDGES/S/(days%20since%201999-01-01)/streamgridunitconvert/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/RANGEEDGES/%5BM%5Daverage/L/'+str(nday)+'/runningAverage/SOURCES/.Models/.SubX/.EMC/.GEFS/.hindcast/.dc9915/.pr/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/RANGEEDGES/L/'+str(nday)+'/runningAverage/S/(T)/renameGRID/pentadmean/T/(S)/renameGRID/%5BS%5DregridLinear/sub/S/('+training_season+')/VALUES/L/removeGRID/S/(T)/renameGRID/c%3A/0.001/(m3%20kg-1)/%3Ac/mul/c%3A/1000/(mm%20m-1)/%3Ac/mul/c%3A/86400/(s%20day-1)/%3Ac/mul/c%3A/7.0//units//days/def/%3Ac/mul/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2301/ensotime/%3Agrid/use_as_grid/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
		}
		# calls curl to download data
		url=dic[model]
		print("\n Hindcasts URL: \n\n "+url)
		get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > model_precip_"+mon+"_wk"+str(week)+".tsv.gz")
		get_ipython().system("gunzip -f model_precip_"+mon+"_wk"+str(week)+".tsv.gz")
		#! curl -g -k -b '__dlauth_id='$key'' ''$url'' > model_precip_${mo}.tsv

def GetHindcasts_Temp(wlo1, elo1, sla1, nla1, day1, day2, fyr, mon, os, key, week, nlag, nday, training_season, hstep, model, hdate_last, force_download):
	if not force_download:
		try:
			ff=open("model_precip_"+mon+"_wk"+str(week)+".tsv", 'r')
			s = ff.readline()
		except OSError as err:
			print("\033[1mWarning:\033[0;0m {0}".format(err))
			print("Hindcasts file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
			force_download = True
	if force_download:
		#dictionary:
		dic = { 'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_temperature/.skt/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/LA/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%201%20'+mon+'%20'+str(fyr)+')%20(2300%2028%20'+mon+'%20'+str(fyr)+')/RANGE/%5BL%5Daverage//Celsius/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(hdate_last)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/T/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2060/ensotime/%3Agrid/replaceGRID//name/(temp)/def//units/(Celsius)/def//long_name/(surface temperature)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
		}
		# calls curl to download data
		url=dic[model]
		print("\n Hindcasts URL: \n\n "+url)
		get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > model_precip_"+mon+"_wk"+str(week)+".tsv.gz")
		get_ipython().system("gunzip -f model_precip_"+mon+"_wk"+str(week)+".tsv.gz")
		#! curl -g -k -b '__dlauth_id='$key'' ''$url'' > model_precip_${mo}.tsv

def GetHindcasts_RFREQ(wlo1, elo1, sla1, nla1, day1, day2, nday, fyr, mon, os, key, week, wetday_threshold, nlag, training_season, hstep,model, force_download):
	nwi=4  #number of weeks to use for real-time ECMWF training period (2 initializations per week) --Send to namelist in the future
	if not force_download:
		try:
			ff=open("model_RFREQ_"+mon+"_wk"+str(week)+".tsv", 'r')
			s = ff.readline()
		except OSError as err:
			print("\033[1mWarning:\033[0;0m {0}".format(err))
			print("Hindcasts file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
			force_download = True
	if force_download:
		#dictionary:
		dic = { 'CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag/M%5Daverage/3./mul/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/add/4./div/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/'+str(hstep)+'/STEP/dup/S/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.S/replaceGRID/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/c://name//water_density/def/998/%28kg/m3%29/:c/div//mm/unitconvert//name/(tp)/def/grid://name/%28T%29/def//units/%28months%20since%201960-01-01%29/def//standard_name/%28time%29/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/:grid/use_as_grid//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
				'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%201%20'+mon+'%20'+str(fyr)+')%20(2300%2028%20'+mon+'%20'+str(fyr)+')/RANGE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/'+str(wetday_threshold)+'/flagge/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(fyr-1)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/T/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2060/ensotime/%3Agrid/replaceGRID//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
				'ECMWFrt': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(last%206%20'+str(nwi)+'%20mul%20sub)%20(last)/RANGE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/'+str(wetday_threshold)+'/flagge/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(fyr-1)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/T/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2060/ensotime/%3Agrid/replaceGRID//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
		}
		# calls curl to download data
		url=dic[model]
		print("\n Hindcasts URL: \n\n "+url)
		get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > model_RFREQ_"+mon+"_wk"+str(week)+".tsv.gz")
		get_ipython().system("gunzip -f model_RFREQ_"+mon+"_wk"+str(week)+".tsv.gz")
		#! curl -g -k -b '__dlauth_id='$key'' ''$url'' > model_precip_${mo}.tsv

def GetHindcastsUser(wlo1, elo1, sla1, nla1, day1, day2, fyr, mon, os, key, week, nlag, nday, training_season, hstep, model, hdate_last, force_download):
	if not force_download:
		try:
			ff=open("model_precip_"+mon+"_wk"+str(week)+".tsv", 'r')
			s = ff.readline()
		except OSError as err:
			print("\033[1mWarning:\033[0;0m {0}".format(err))
			print("Hindcasts file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
			force_download = True
	if force_download:
		#dictionary:
#		dic = { 'CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag/M%5Daverage/3./mul/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/add/4./div/S/('+training_season+')/VALUES/S/'+str(hstep)+'/STEP/dup/S/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.S/replaceGRID/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/c://name//water_density/def/998/%28kg/m3%29/:c/div//mm/unitconvert//name/(tp)/def/grid://name/%28T%29/def//units/%28months%20since%201960-01-01%29/def//standard_name/%28time%29/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/:grid/use_as_grid//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
		dic = { 'NextGenR0': 'http://iridl.ldeo.columbia.edu/home/.xchourio/.IRAP2/.S2S/.R0HIND/.Caminadeetal/home/.xchourio/.IRAP2/.S2S/.R0HIND/.Mordecaietal/add/home/.xchourio/.IRAP2/.S2S/.R0HIND/.Wesolowskietal/add/home/.xchourio/.IRAP2/.S2S/.R0HIND/.LiuHelmerssonetal/add/4/div/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/RANGE/%5BL%5D//keepgrids/average/%5BM%5Daverage/S/%28days%20since%201960-01-01%29/streamgridunitconvert/%5BX/Y%5DREORDER/2/RECHUNK/S//pointwidth/0/def/30/shiftGRID/S//units//days/def/L/add/0/RECHUNK//name//T/def//long_name/%28Target%20date%29/def/2/%7Bexch%5BS/L%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/T/grid://name/%28T%29/def//units/%28months%20since%201960-01-01%29/def//standard_name/%28time%29/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1920/ensotime/:grid/replaceGRID//name/%28R0%29/def//units/%28unitless%29/def//long_name/%28R0%29/def/-999/replaceNaN/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
		}
		# calls curl to download data
		url=dic[model]
		print("\n Hindcasts URL: \n\n "+url)
		get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > model_precip_"+mon+"_wk"+str(week)+".tsv.gz")
		get_ipython().system("gunzip -f model_precip_"+mon+"_wk"+str(week)+".tsv.gz")
		#! curl -g -k -b '__dlauth_id='$key'' ''$url'' > model_precip_${mo}.tsv

def GetObs(day1, day2, mon, fyr, wlo2, elo2, sla2, nla2, nday, key, week, nlag, training_season, hstep, model, obs_source, obsclimo_source, hdate_last, force_download):
	nwi=4  #number of weeks to use for real-time ECMWF training period (2 initializations per week) --Send to namelist in the future
	if not force_download:
		try:
			ff=open("obs_precip_"+mon+"_wk"+str(week)+".tsv", 'r')
			s = ff.readline()
		except OSError as err:
			print("\033[1mWarning:\033[0;0m {0}".format(err))
			print("Obs precip file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
			force_download = True
	if force_download:
		#dictionary:
#		dic = {'CFSv2':                'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/'+str(hstep)+'/STEP/L1/S/add/0/RECHUNK/name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/3/flagge/dup/pentadmean/%5BT%5D/regridLinear/sub/T/'+str(nday)+'/runningAverage/c%3A/7.0//units//days/def/%3Ac/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/I/3/-1/roll/.T/replaceGRID/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/%3Agrid/use_as_grid/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
		dic = {'CFSv2':                'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+mon+')/VALUES/S/'+str(hstep)+'/STEP/L1/S/add/0/RECHUNK/name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/(T)/use_as_grid/'+obs_source+'/T/(days%20since%201960-01-01)/streamgridunitconvert/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/'+obsclimo_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/%5BT%5D/regridLinear/sub/T/'+str(nday)+'/runningAverage/c%3A/7.0//units//days/def/%3Ac/mul/T/2/index/.T/SAMPLE/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/%3Agrid/use_as_grid/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
			   'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%201%20'+mon+'%20'+str(fyr)+')%20(2300%2028%20'+mon+'%20'+str(fyr)+')/RANGE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(hdate_last)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/dup%5BT%5Daverage/sub/-999/setmissing_value/nip/T/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2060/ensotime/%3Agrid/replaceGRID//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
			   'ECMWFrt': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(last%206%20'+str(nwi)+'%20mul%20sub)%20(last)/RANGE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(hdate_last)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/dup%5BT%5Daverage/sub/-999/setmissing_value/nip/T/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2060/ensotime/%3Agrid/replaceGRID//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
			   'GEFS':       'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.EMC/.GEFS/.hindcast/.pr/S/(0000%206%20Jan%201999)/(0000%2028%20Dec%202015)/RANGEEDGES/L/('+str(day1)+')/('+str(day2)+')/RANGEEDGES/L/'+str(nday)+'/runningAverage/S/('+training_season+')/VALUES/L/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/dup/pentadmean/%5BT%5D/regridLinear/sub/T/'+str(nday)+'/runningAverage/c%3A/7.0//units//days/def/%3Ac/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/I/3/-1/roll/.T/replaceGRID/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2301/ensotime/%3Agrid/use_as_grid/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz'
			  }
		# calls curl to download data
		url=dic[model]
		print("\n Obs (Rainfall) data URL: \n\n "+url)
		get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > obs_precip_"+mon+"_wk"+str(week)+".tsv.gz")
		get_ipython().system("gunzip -f obs_precip_"+mon+"_wk"+str(week)+".tsv.gz")
		#curl -g -k -b '__dlauth_id='$key'' ''$url'' > obs_precip_${mo}.tsv

def GetObsTn(day1, day2, mon, fyr, wlo2, elo2, sla2, nla2, nday, key, week, nlag, training_season, hstep, model, obs_source, hdate_last, force_download):
	if not force_download:
		try:
			ff=open("obs_tmin_"+mon+"_wk"+str(week)+".tsv", 'r')
			s = ff.readline()
		except OSError as err:
			print("\033[1mWarning:\033[0;0m {0}".format(err))
			print("Obs temp file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
			force_download = True
	if force_download:
		#dictionary:
		dic = {'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%201%20'+mon+'%20'+str(fyr)+')%20(2300%2028%20'+mon+'%20'+str(fyr)+')/RANGE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(hdate_last)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/dup%5BT%5Daverage/sub/-999/setmissing_value/nip/T/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2060/ensotime/%3Agrid/replaceGRID//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
			  }
		# calls curl to download data
		url=dic[model]
		print("\n Obs (Temp min) data URL: \n\n "+url)
		get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > obs_tmin_"+mon+"_wk"+str(week)+".tsv.gz")
		get_ipython().system("gunzip -f obs_tmin_"+mon+"_wk"+str(week)+".tsv.gz")
		#curl -g -k -b '__dlauth_id='$key'' ''$url'' > obs_precip_${mo}.tsv

def GetObs_RFREQ(day1, day2, mon, fyr, wlo2, elo2, sla2, nla2, nday, key, week, wetday_threshold, threshold_pctle, nlag, training_season, hstep, model, obs_source, force_download):
	nwi=4  #number of weeks to use for real-time ECMWF training period (2 initializations per week) --Send to namelist in the future
	if not force_download:
		try:
			ff=open("obs_RFREQ_"+mon+"_wk"+str(week)+".tsv", 'r')
			s = ff.readline()
		except OSError as err:
			print("\033[1mWarning:\033[0;0m {0}".format(err))
			print("Obs freq-rainfall file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
			force_download = True
	if force_download:
		#dictionaries:
		if threshold_pctle:
				dic = { 'CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/'+str(hstep)+'/STEP/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/T/(1%20Jan%201999)/(31%20Dec%202011)/RANGEEDGES/%5BT%5Dpercentileover/'+str(wetday_threshold)+'/flagle/T/'+str(nday)+'/runningAverage/'+str(nday)+'/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/%3Agrid/use_as_grid//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
						'ECMWF':'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%201%20'+mon+'%20'+str(fyr)+')%20(2300%2028%20'+mon+'%20'+str(fyr)+')/RANGE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(fyr-1)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/%5BT%5Dpercentileover/'+str(wetday_threshold)+'/flagle/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2060/ensotime/%3Agrid/use_as_grid//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
						'ECMWFrt':'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(last%206%20'+str(nwi)+'%20mul%20sub)%20(last)/RANGE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(fyr-1)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/%5BT%5Dpercentileover/'+str(wetday_threshold)+'/flagle/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2060/ensotime/%3Agrid/use_as_grid//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
						}
		else:
				dic = { 'CFSv2':                     'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/'+str(hstep)+'/STEP/L1/S/add/0/RECHUNK/name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/'+str(wetday_threshold)+'/flagge/dup/pentadmean/%5BT%5D/regridLinear/sub/T/'+str(nday)+'/runningAverage/c%3A/7.0//units//days/def/%3Ac/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/I/3/-1/roll/.T/replaceGRID/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/%3Agrid/use_as_grid/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
						'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%201%20'+mon+'%20'+str(fyr)+')%20(2300%2028%20'+mon+'%20'+str(fyr)+')/RANGE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(fyr-1)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/'+str(wetday_threshold)+'/flagge/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2060/ensotime/%3Agrid/use_as_grid//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
						'ECMWFrt': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(last%206%20'+str(nwi)+'%20mul%20sub)%20(last)/RANGE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(fyr-1)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/'+str(wetday_threshold)+'/flagge/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2060/ensotime/%3Agrid/use_as_grid//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
						'GEFS':       'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.EMC/.GEFS/.hindcast/.pr/S/(0000%206%20Jan%201999)/(0000%2028%20Dec%202015)/RANGEEDGES/L/('+str(day1)+')/('+str(day2)+')/RANGEEDGES/L/'+str(nday)+'/runningAverage/S/('+training_season+')/VALUES/L/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.NASA/.GES-DAAC/.TRMM_L3/.TRMM_3B42/.v7/.daily/.precipitation/X/0./1.5/360./GRID/Y/-50/1.5/50/GRID/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/'+str(wetday_threshold)+'/flagge/dup/pentadmean/%5BT%5D/regridLinear/sub/T/'+str(nday)+'/runningAverage/c%3A/7.0//units//days/def/%3Ac/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/I/3/-1/roll/.T/replaceGRID/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2301/ensotime/%3Agrid/use_as_grid/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz'
						}
		# calls curl to download data
		url=dic[model]
		print("\n Obs (Freq) data URL: \n\n "+url)
		get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > obs_RFREQ_"+mon+"_wk"+str(week)+".tsv.gz")
		get_ipython().system("gunzip -f obs_RFREQ_"+mon+"_wk"+str(week)+".tsv.gz")
		#curl -g -k -b '__dlauth_id='$key'' ''$url'' > obs_precip_${mo}.tsv

def GetObsUser(day1, day2, mon, fyr, wlo2, elo2, sla2, nla2, nday, key, week, nlag, training_season, hstep, model, obs_source, obsclimo_source, hdate_last, force_download):
	if not force_download:
		try:
			ff=open("obs_precip_"+mon+"_wk"+str(week)+".tsv", 'r')
			s = ff.readline()
		except OSError as err:
			print("\033[1mWarning:\033[0;0m {0}".format(err))
			print("Obs precip file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
			force_download = True
	if force_download:
		#dictionary:
#		dic = {'CFSv2':                'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/'+str(hstep)+'/STEP/L1/S/add/0/RECHUNK/name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/3/flagge/dup/pentadmean/%5BT%5D/regridLinear/sub/T/'+str(nday)+'/runningAverage/c%3A/7.0//units//days/def/%3Ac/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/I/3/-1/roll/.T/replaceGRID/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/%3Agrid/use_as_grid/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
		dic = {'NextGenR0': 'http://iridl.ldeo.columbia.edu/home/.xchourio/.IRAP2/.S2S/.R0HIND/.Caminadeetal/home/.xchourio/.IRAP2/.S2S/.R0HIND/.Mordecaietal/add/home/.xchourio/.IRAP2/.S2S/.R0HIND/.Wesolowskietal/add/home/.xchourio/.IRAP2/.S2S/.R0HIND/.LiuHelmerssonetal/add/4/div/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/RANGE/%5BL%5D//keepgrids/average/%5BM%5Daverage/S/(days%20since%201960-01-01)/streamgridunitconvert/%5BX/Y%5DREORDER/2/RECHUNK/S//pointwidth/0/def/30/shiftGRID/S//units//days/def/L/add/0/RECHUNK//name//T/def//long_name/(Target%20date)/def/2/%7Bexch%5BS/L%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/home/.xchourio/.IRAP2/.S2S/.R0OBS/.Caminadeetal/home/.xchourio/.IRAP2/.S2S/.R0OBS/.Mordecaietal/add/home/.xchourio/.IRAP2/.S2S/.R0OBS/.Wesolowskietal/add/home/.xchourio/.IRAP2/.S2S/.R0OBS/.LiuHelmerssonetal/add/4/div/T/2/index/.T/SAMPLE/T/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1920/ensotime/%3Agrid/replaceGRID//name/(R0%20obs)/def//units/(unitless)/def//long_name/(R0)/def/-999/replaceNaN/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
		}
		# calls curl to download data
		url=dic[model]
		print("\n R0 'obs' data URL: \n\n "+url)
		get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > obs_precip_"+mon+"_wk"+str(week)+".tsv.gz")
		get_ipython().system("gunzip -f obs_precip_"+mon+"_wk"+str(week)+".tsv.gz")
		#curl -g -k -b '__dlauth_id='$key'' ''$url'' > obs_precip_${mo}.tsv

def GetForecastUser(day1, day2, fday, mon, fyr, nday, wlo1, elo1, sla1, nla1, wlo2, elo2, sla2, nla2, obs_source, key, week, nlag, model, hdate_last, threshold_pctle,training_season,wetday_threshold,force_download,mpref):
	if not force_download:
		try:
			ff=open("modelfcst_precip_"+mon+"_fday"+str(fday)+"_wk"+str(week)+".tsv", 'r')
			s = ff.readline()
		except OSError as err:
			print("\033[1mWarning:\033[0;0m {0}".format(err))
			print("Forecasts file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
			force_download = True
	if force_download:
		#dictionary:
		dic = { 'NextGenR0': 'http://iridl.ldeo.columbia.edu/home/.xchourio/.IRAP2/.S2S/.R0HIND/.Caminadeetal/home/.xchourio/.IRAP2/.S2S/.R0HIND/.Mordecaietal/add/home/.xchourio/.IRAP2/.S2S/.R0HIND/.Wesolowskietal/add/home/.xchourio/.IRAP2/.S2S/.R0HIND/.LiuHelmerssonetal/add/4/div/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/RANGE/%5BL%5D//keepgrids/average/%5BM%5Daverage/S/%28days%20since%201960-01-01%29/streamgridunitconvert/%5BX/Y%5DREORDER/2/RECHUNK/S//pointwidth/0/def/30/shiftGRID/S//units//days/def/L/add/0/RECHUNK//name//T/def//long_name/%28Target%20date%29/def/2/%7Bexch%5BS/L%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/T/grid://name/%28T%29/def//units/%28months%20since%201960-01-01%29/def//standard_name/%28time%29/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1920/ensotime/:grid/replaceGRID//name/%28R0%29/def//units/%28unitless%29/def//long_name/%28R0%29/def/-999/replaceNaN/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
		}
		# calls curl to download data
		url=dic[model]
		print("\n Forecast URL: \n\n "+url)
		get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > modelfcst_precip_"+mon+"_fday"+str(fday)+"_wk"+str(week)+".tsv.gz")
		get_ipython().system("gunzip -f modelfcst_precip_"+mon+"_fday"+str(fday)+"_wk"+str(week)+".tsv.gz")
		#curl -g -k -b '__dlauth_id='$key'' ''$url'' > modelfcst_precip_fday${fday}.tsv

def GetForecast(day1, day2, fday, mon, fyr, nday, wlo1, elo1, sla1, nla1, wlo2, elo2, sla2, nla2, obs_source,obsclimo_source, key, week, nlag, model, hdate_last, threshold_pctle,training_season,wetday_threshold,force_download,mpref):
	if not force_download:
		try:
			ff=open("modelfcst_precip_"+mon+"_fday"+str(fday)+"_wk"+str(week)+".tsv", 'r')
			s = ff.readline()
		except OSError as err:
			print("\033[1mWarning:\033[0;0m {0}".format(err))
			print("Forecasts file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
			force_download = True
	if force_download:
		#dictionary:
		dic = {	'CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag/M%5Daverage/3.0/mul/SOURCES/.ECMWF/.S2S/.NCEP/.forecast/.control/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/add/4.0/div/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')VALUE/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag/M%5Daverage/3.0/mul/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/add/4.0/div/S/(0000%20'+str(fday)+'%20'+mon+')VALUES/%5BS%5Daverage/sub/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/1/Jan/3001/ensotime/12.0/1/Jan/3001/ensotime/%3Agrid/addGRID/T//pointwidth/0/def/pop//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
				'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/%5BM%5Daverage/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/%5BM%5Daverage/%5Bhdate%5Daverage/sub/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/1/Jan/3001/ensotime/12.0/1/Jan/3001/ensotime/%3Agrid/addGRID/T//pointwidth/0/def/pop//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
				'ECMWFrt': 'http://iridl.ldeo.columbia.edu/home/.jingyuan/.ECMWF/.realtime_S2S/.ECMF/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(last)/VALUE/%5BL%5Ddifferences/%5BM%5Daverage/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/%5BM%5Daverage/%5Bhdate%5Daverage/sub/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/1/Jan/3001/ensotime/12.0/1/Jan/3001/ensotime/%3Agrid/addGRID/T//pointwidth/0/def/pop//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
				'GEFS':  'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.EMC/.GEFS/.forecast/.pr/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUES/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/RANGEEDGES/%5BM%5Daverage/L/'+str(nday)+'/runningAverage/SOURCES/.Models/.SubX/.EMC/.GEFS/.hindcast/.dc9915/.pr/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/RANGEEDGES/L/'+str(nday)+'/runningAverage/S/(T)/renameGRID/pentadmean/T/(S)/renameGRID/%5BS%5DregridLinear/S/1/setgridtype/pop/S/2/index/.S/SAMPLE/sub/c%3A/0.001/(m3%20kg-1)/%3Ac/mul/c%3A/1000/(mm%20m-1)/%3Ac/mul/c%3A/86400/(s%20day-1)/%3Ac/mul/c%3A/7.0//units//days/def/%3Ac/mul/S/(T)/renameGRID/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/3001/ensotime/12.0/16/Jan/3001/ensotime/%3Agrid/use_as_grid/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz'
			  }
		# calls curl to download data
		url=dic[model]
		print("\n Forecast URL: \n\n "+url)
		get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > modelfcst_precip_"+mon+"_fday"+str(fday)+"_wk"+str(week)+".tsv.gz")
		get_ipython().system("gunzip -f modelfcst_precip_"+mon+"_fday"+str(fday)+"_wk"+str(week)+".tsv.gz")
		#curl -g -k -b '__dlauth_id='$key'' ''$url'' > modelfcst_precip_fday${fday}.tsv

	#False force_download
	if mpref=='noMOS':
		force_download = False

		#The next two if-blocks are used for noMOS forecasts ##Added by AGM
		#Short hindcast to correctly compute climatological period of the actual forecast
		if not force_download:
			try:
				ff=open("noMOS/modelshort_precip_"+mon+"_wk"+str(week)+".tsv", 'r')
				s = ff.readline()
			except OSError as err:
				print("\033[1mWarning:\033[0;0m {0}".format(err))
				print("Short hindcast file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
				force_download = True
		if force_download:
			#dictionary:
			dic = { 'CFSv2': 'https://iridl.ldeo.columbia.edu/home/.xchourio/.S2SDB/.NCEP/.reforecasts/.perturbed/.anomalies/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag/M%5Daverage/3./mul/home/.xchourio/.S2SDB/.NCEP/.reforecasts/.control/.anomalies/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/add/4./div/S/(0000%20'+str(fday)+'%20'+mon+')/VALUES/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/c://name//water_density/def/998/%28kg/m3%29/:c/div//mm/unitconvert//name/(tp)/def/grid://name/%28T%29/def//units/%28months%20since%201960-01-01%29/def//standard_name/%28time%29/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/:grid/use_as_grid//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
					'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(fyr-1)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/T/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1920/ensotime/%3Agrid/replaceGRID//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
					'ECMWFrt': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(last)/VALUE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(fyr-1)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/T/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1920/ensotime/%3Agrid/replaceGRID//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
			}
			# calls curl to download data
			url=dic[model]
			print("\n Short hindcast file\n") #URL: \n\n "+url)
			get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > noMOS/modelshort_precip_"+mon+"_wk"+str(week)+".tsv.gz")
			get_ipython().system("gunzip -f noMOS/modelshort_precip_"+mon+"_wk"+str(week)+".tsv.gz")
		#Short obs period corresponding to the short hindcast period

		#False force_download
		force_download = False

		if not force_download:
			try:
				ff=open("noMOS/obsshort_precip_"+mon+"_wk"+str(week)+".tsv", 'r')
				s = ff.readline()
			except OSError as err:
				print("\033[1mWarning:\033[0;0m {0}".format(err))
				print("Short obs precip file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
				force_download = True
		if force_download:
			#dictionary:
			dic = {'CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/(0000%20'+str(fday)+'%20'+mon+')/VALUES/L1/S/add/0/RECHUNK/name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/(T)/use_as_grid/'+obs_source+'/T/(days%20since%201960-01-01)/streamgridunitconvert/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/'+obsclimo_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/%5BT%5D/regridLinear/sub/T/'+str(nday)+'/runningAverage/c%3A/7.0//units//days/def/%3Ac/mul/T/2/index/.T/SAMPLE/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/%3Agrid/use_as_grid/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
				   'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(fyr-1)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/dup%5BT%5Daverage/sub/-999/setmissing_value/nip/T/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1920/ensotime/%3Agrid/replaceGRID//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
				   'ECMWFrt': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(last)/VALUE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(fyr-1)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/dup%5BT%5Daverage/sub/-999/setmissing_value/nip/T/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1920/ensotime/%3Agrid/replaceGRID//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
				   }
			# calls curl to download data
			url=dic[model]
			print("\n Short obs (Rainfall) data URL: \n\n "+url)
			get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > noMOS/obsshort_precip_"+mon+"_wk"+str(week)+".tsv.gz")
			get_ipython().system("gunzip -f noMOS/obsshort_precip_"+mon+"_wk"+str(week)+".tsv.gz")

		#False force_download
		force_download = False

		#The next block is used for noMOS probabilistic forecasts ##Added by AGM
		#Above normal:
		if not force_download:
			try:
				ff=Dataset('noMOS/modelfcst_above_PRCP_'+mon+'_wk'+str(week)+'.nc', 'r')
				s = ff.variables['Y'][:]
			except OSError as err:
				print("\033[1mWarning:\033[0;0m {0}".format(err))
				print("Above normal probability forecast file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
				force_download = True
		if force_download:
			#dictionary:
			dic = { 'CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L1/('+str(day1)+')/('+str(day2)+')/VALUES%5BL1%5Ddifferences/SOURCES/.ECMWF/.S2S/.NCEP/.forecast/.control/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L1/('+str(day1)+')/('+str(day2)+')/VALUES%5BL1%5Ddifferences//M//ids/ordered%5B16%5DNewGRID/addGRID/appendstream%5BX/Y/M/L1/S%5DREORDER/2/RECHUNK/S/-2/1/0/shiftdatashort/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE%5BX/Y/L1/S/M/S_lag%5DREORDER/4/RECHUNK%5BM/S_lag%5D//M1/nchunk/NewIntegerGRID/replaceGRIDstream/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L1/('+str(day1)+')/('+str(day2)+')/VALUES%5BL1%5Ddifferences/S/-2/1/0/shiftdatashort/S/(0000%20'+str(fday)+'%20'+mon+')/VALUES%5BX/Y/L1/M/S_lag/S%5DREORDER/3/RECHUNK%5BM/S_lag/S%5D//M2/nchunk/NewIntegerGRID/replaceGRIDstream%5BM2%5D0.33/0.66/0/replacebypercentile/percentile/0.67/VALUE/flaggt%5BM1%5Daverage/100/mul//long_name/%28Probability%20of%20Above%20Normal%20Tercile%29def//units/%28%25%29def/data.nc',
				    'ECMWF': 'http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/hdate/('+str(fyr-20)+')/('+str(hdate_last)+')/RANGE%5BL%5Ddifferences%5BM%5Daverage%5Bhdate%5D0.33/0.66/0/replacebypercentile/percentile/0.66/VALUE/flaggt%5BM%5Daverage/100/mul//long_name/%28Probability%20of%20Above%20Normal%20Tercile%29def//units/%28%25%29def/data.nc',
				    'ECMWFrt': 'http://iridl.ldeo.columbia.edu/home/.jingyuan/.ECMWF/.realtime_S2S/.ECMF/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(last)/VALUE/%5BL%5Ddifferences/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/hdate/('+str(fyr-20)+')/('+str(hdate_last)+')/RANGE%5BL%5Ddifferences%5BM%5Daverage%5Bhdate%5D0.33/0.66/0/replacebypercentile/percentile/0.66/VALUE/flaggt%5BM%5Daverage/100/mul//long_name/%28Probability%20of%20Above%20Normal%20Tercile%29def//units/%28%25%29def/data.nc',
			}
			# calls curl to download data
			url=dic[model]
			print("\n Short hindcast URL: \n\n "+url)
			get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > noMOS/modelfcst_above_PRCP_"+mon+"_wk"+str(week)+".nc")

		#False force_download
		force_download = False

		#Below normal:
		if not force_download:
			try:
				ff=Dataset("noMOS/modelfcst_below_PRCP_"+mon+"_wk"+str(week)+".nc", 'r')
				s = ff.variables['Y'][:]
			except OSError as err:
				print("\033[1mWarning:\033[0;0m {0}".format(err))
				print("Below normal probability forecast file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
				force_download = True
		if force_download:
			#dictionary:
			dic = { 'CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L1/('+str(day1)+')/('+str(day2)+')/VALUES%5BL1%5Ddifferences/SOURCES/.ECMWF/.S2S/.NCEP/.forecast/.control/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L1/('+str(day1)+')/('+str(day2)+')/VALUES%5BL1%5Ddifferences//M//ids/ordered%5B16%5DNewGRID/addGRID/appendstream%5BX/Y/M/L1/S%5DREORDER/2/RECHUNK/S/-2/1/0/shiftdatashort/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE%5BX/Y/L1/S/M/S_lag%5DREORDER/4/RECHUNK%5BM/S_lag%5D//M1/nchunk/NewIntegerGRID/replaceGRIDstream/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L1/('+str(day1)+')/('+str(day2)+')/VALUES%5BL1%5Ddifferences/S/-2/1/0/shiftdatashort/S/(0000%20'+str(fday)+'%20'+mon+')/VALUES%5BX/Y/L1/M/S_lag/S%5DREORDER/3/RECHUNK%5BM/S_lag/S%5D//M2/nchunk/NewIntegerGRID/replaceGRIDstream%5BM2%5D0.33/0.66/0/replacebypercentile/percentile/0.33/VALUE/flaglt%5BM1%5Daverage/100/mul//long_name/%28Probability%20of%20Below%20Normal%20Tercile%29def//units/%28%25%29def/data.nc',
				'ECMWF': 'http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/hdate/('+str(fyr-20)+')/('+str(hdate_last)+')/RANGE%5BL%5Ddifferences%5BM%5Daverage%5Bhdate%5D0.33/0.66/0/replacebypercentile/percentile/0.33/VALUE/flaglt%5BM%5Daverage/100/mul//long_name/%28Probability%20of%20Below%20Normal%20Tercile%29def//units/%28%25%29def/data.nc',
				'ECMWFrt': 'http://iridl.ldeo.columbia.edu/home/.jingyuan/.ECMWF/.realtime_S2S/.ECMF/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(last)/VALUE/%5BL%5Ddifferences/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/hdate/('+str(fyr-20)+')/('+str(hdate_last)+')/RANGE%5BL%5Ddifferences%5BM%5Daverage%5Bhdate%5D0.33/0.66/0/replacebypercentile/percentile/0.33/VALUE/flaglt%5BM%5Daverage/100/mul//long_name/%28Probability%20of%20Below%20Normal%20Tercile%29def//units/%28%25%29def/data.nc',
			}
			# calls curl to download data
			url=dic[model]
			print("\n Short hindcast URL: \n\n "+url)
			get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > noMOS/modelfcst_below_PRCP_"+mon+"_wk"+str(week)+".nc")

		#False force_download
		force_download = False

		#The next block is used for noMOS flexible probabilistic forecasts ##Added by AGM
		#Ensemble mean:
		if not force_download:
			try:
				ff=Dataset('noMOS/modelfcst_mu_PRCP_'+mon+'_wk'+str(week)+'.nc', 'r')
				s = ff.variables['Y'][:]
			except OSError as err:
				print("\033[1mWarning:\033[0;0m {0}".format(err))
				print("Ensemble mean file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
				force_download = True
		if force_download:
			#dictionary:
			dic = { 'CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag/M%5Daverage/3.0/mul/SOURCES/.ECMWF/.S2S/.NCEP/.forecast/.control/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/add/4.0/div/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')VALUE/home/.xchourio/.S2SDB/.NCEP/.reforecasts/.climatologies/.sfc_precip/.tpSmooth/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/S/(0000%20'+str(fday)+'%20'+mon+')VALUES/sub/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/data.nc',
			        'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/%5BM%5Daverage/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/%5BM%5Daverage/%5Bhdate%5Daverage/sub/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/data.nc',
			        'ECMWFrt': 'http://iridl.ldeo.columbia.edu/home/.jingyuan/.ECMWF/.realtime_S2S/.ECMF/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(last)/VALUE/%5BL%5Ddifferences/%5BM%5Daverage/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/%5BM%5Daverage/%5Bhdate%5Daverage/sub/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/data.nc',
			}
			# calls curl to download data
			url=dic[model]
			print("\n Ensemble mean URL: \n\n "+url)
			get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > noMOS/modelfcst_mu_PRCP_"+mon+"_wk"+str(week)+".nc")

		#False force_download
		force_download = False

		#Ensemble standard deviation:
		if not force_download:
			try:
				ff=Dataset("noMOS/modelfcst_std_PRCP_"+mon+"_wk"+str(week)+".nc", 'r')
				s = ff.variables['Y'][:]
			except OSError as err:
				print("\033[1mWarning:\033[0;0m {0}".format(err))
				print("Ensemble standard deviation file file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
				force_download = True
		if force_download:
			#dictionary:
			dic = { 'CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L1/('+str(day1)+')/('+str(day2)+')/VALUES%5BL1%5Ddifferences/SOURCES/.ECMWF/.S2S/.NCEP/.forecast/.control/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L1/('+str(day1)+')/('+str(day2)+')/VALUES%5BL1%5Ddifferences//M//ids/ordered%5B16%5DNewGRID/addGRID/appendstream%5BX/Y/M/L1/S%5DREORDER/2/RECHUNK/S/-2/1/0/shiftdatashort/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE%5BX/Y/L1/S/M/S_lag%5DREORDER/4/RECHUNK%5BM/S_lag%5D//M1/nchunk/NewIntegerGRID/replaceGRIDstream/%5BM1%5Drmsover/c://name//water_density/def/998/%28kg/m3%29/:c/div//mm/unitconvert/data.nc',
			    'ECMWF': 'http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/%5BM%5Drmsover/c://name//water_density/def/998/%28kg/m3%29/:c/div//mm/unitconvert/data.nc',
				'ECMWFrt': 'http://iridl.ldeo.columbia.edu/home/.jingyuan/.ECMWF/.realtime_S2S/.ECMF/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(last)/VALUE/%5BL%5Ddifferences/%5BM%5Drmsover/c://name//water_density/def/998/%28kg/m3%29/:c/div//mm/unitconvert/data.nc',
			}
			# calls curl to download data
			url=dic[model]
			#print("\n Ensemble std URL: \n\n "+url)
			get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > noMOS/modelfcst_std_PRCP_"+mon+"_wk"+str(week)+".nc")

		#Obs mean:
		if not force_download:
			try:
				ff=Dataset('noMOS/obs_mu_PRCP_'+mon+'_wk'+str(week)+'.nc', 'r')
				s = ff.variables['Y'][:]
			except OSError as err:
				print("\033[1mWarning:\033[0;0m {0}".format(err))
				print("Obs mean file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
				force_download = True
		if force_download:
			#dictionary:
			dic = { 'CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/(0000%20'+str(fday)+'%20'+mon+')/VALUES/L1/S/add/0/RECHUNK/name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/(T)/use_as_grid/'+obs_source+'/T/(days%20since%201960-01-01)/streamgridunitconvert/%5BX/Y%5D/regridAverage/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/'+obsclimo_source+'/%5BX/Y%5D/regridAverage/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/%5BT%5D/regridLinear/sub/T/'+str(nday)+'/runningAverage/c%3A/7.0//units//days/def/%3Ac/mul/T/2/index/.T/SAMPLE/nip/%5BT%5Daverage//name/(tp)/def/data.nc',
			        'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(hdate_last)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/%5BX/Y%5D/regridAverage/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/dup%5BT%5Daverage/sub/-999/setmissing_value/nip/T/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1920/ensotime/%3Agrid/replaceGRID//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/%5BT%5Daverage/data.nc',
			        'ECMWFrt': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(last)/VALUE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(hdate_last)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/%5BX/Y%5D/regridAverage/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/dup%5BT%5Daverage/sub/-999/setmissing_value/nip/T/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1920/ensotime/%3Agrid/replaceGRID//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/%5BT%5Daverage/data.nc',
			}
			# calls curl to download data
			url=dic[model]
			#print("\n Obs mean URL: \n\n "+url)
			get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > noMOS/obs_mu_PRCP_"+mon+"_wk"+str(week)+".nc")

		#False force_download
		force_download = False

		#Obs std:
		if not force_download:
			try:
				ff=Dataset('noMOS/obs_std_PRCP_'+mon+'_wk'+str(week)+'.nc', 'r')
				s = ff.variables['Y'][:]
			except OSError as err:
				print("\033[1mWarning:\033[0;0m {0}".format(err))
				print("Obs std file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
				force_download = True
		if force_download:
			#dictionary:
			dic = { 'CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/(0000%20'+str(fday)+'%20'+mon+')/VALUES/L1/S/add/0/RECHUNK/name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/(T)/use_as_grid/'+obs_source+'/T/(days%20since%201960-01-01)/streamgridunitconvert/%5BX/Y%5D/regridAverage/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/'+obsclimo_source+'/%5BX/Y%5D/regridAverage/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/%5BT%5D/regridLinear/sub/T/'+str(nday)+'/runningAverage/c%3A/7.0//units//days/def/%3Ac/mul/T/2/index/.T/SAMPLE/nip/%5BT%5Drmsover//name/(tp)/def/data.nc',
			      'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(hdate_last)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/%5BX/Y%5D/regridAverage/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/-999/setmissing_value/nip/T/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1920/ensotime/%3Agrid/replaceGRID//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/%5BT%5Drmsover/data.nc',
			      'ECMWFrt': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(last)/VALUE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(hdate_last)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/%5BX/Y%5D/regridAverage/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/-999/setmissing_value/nip/T/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1920/ensotime/%3Agrid/replaceGRID//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/%5BT%5Drmsover/data.nc',
			}
			# calls curl to download data
			url=dic[model]
			#print("\n Obs std URL: \n\n "+url)
			get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > noMOS/obs_std_PRCP_"+mon+"_wk"+str(week)+".nc")

		#False force_download
		force_download = False

		if not force_download:
			try:
				ff=open("noMOS/obsshort_RFREQ_"+mon+"_wk"+str(week)+".tsv", 'r')
				s = ff.readline()
			except OSError as err:
				print("\033[1mWarning:\033[0;0m {0}".format(err))
				print("Short obs precip file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
				force_download = True
		#dictionaries:
		if threshold_pctle:
			dic = { 'CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/(0000%20'+str(fday)+'%20'+mon+')/VALUES/L1/S/add/0/RECHUNK/name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/(T)/use_as_grid/'+obs_source+'/T/(days%20since%201960-01-01)/streamgridunitconvert/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/'+obsclimo_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/%5BT%5D/regridLinear/sub/T/'+str(nday)+'/runningAverage/c%3A/7.0//units//days/def/%3Ac/mul/T/2/index/.T/SAMPLE/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/%3Agrid/use_as_grid/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
					'ECMWF':'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(fyr-1)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/%5BT%5Dpercentileover/'+str(wetday_threshold)+'/flagle/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1920/ensotime/%3Agrid/use_as_grid//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
					'ECMWFrt':'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(last)/VALUE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(fyr-1)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/%5BT%5Dpercentileover/'+str(wetday_threshold)+'/flagle/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1920/ensotime/%3Agrid/use_as_grid//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
					}
		else:
			dic = { 'CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/(0000%20'+str(fday)+'%20'+mon+')/VALUES/L1/S/add/0/RECHUNK/name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/(T)/use_as_grid/'+obs_source+'/T/(days%20since%201960-01-01)/streamgridunitconvert/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/'+obsclimo_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/%5BT%5D/regridLinear/sub/T/'+str(nday)+'/runningAverage/c%3A/7.0//units//days/def/%3Ac/mul/T/2/index/.T/SAMPLE/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/%3Agrid/use_as_grid/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
					'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(fyr-1)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/'+str(wetday_threshold)+'/flagge/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1920/ensotime/%3Agrid/use_as_grid//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
					'ECMWFrt': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(last)/VALUE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(fyr-1)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/'+str(wetday_threshold)+'/flagge/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1920/ensotime/%3Agrid/use_as_grid//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
					#'GEFS':       'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.EMC/.GEFS/.hindcast/.pr/S/(0000%206%20Jan%201999)/(0000%2028%20Dec%202015)/RANGEEDGES/L/('+str(day1)+')/('+str(day2)+')/RANGEEDGES/L/'+str(nday)+'/runningAverage/S/('+training_season+')/VALUES/L/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.NASA/.GES-DAAC/.TRMM_L3/.TRMM_3B42/.v7/.daily/.precipitation/X/0./1.5/360./GRID/Y/-50/1.5/50/GRID/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/'+str(wetday_threshold)+'/flagge/dup/pentadmean/%5BT%5D/regridLinear/sub/T/'+str(nday)+'/runningAverage/c%3A/7.0//units//days/def/%3Ac/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/I/3/-1/roll/.T/replaceGRID/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2301/ensotime/%3Agrid/use_as_grid/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz'
					}
			# calls curl to download data
			url=dic[model]
			print("\n Short hindcast file\n") #URL: \n\n "+url)
			get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > noMOS/obsshort_RFREQ_"+mon+"_wk"+str(week)+".tsv.gz")
			get_ipython().system("gunzip -f noMOS/obsshort_RFREQ_"+mon+"_wk"+str(week)+".tsv.gz")
	else:
		print("Data download of forecast individual ensemble members skipped for MOS case")

def GetForecast_RFREQ(day1, day2, fday, mon, fyr, nday, wlo1, elo1, sla1, nla1, wlo2, elo2, sla2, nla2, obs_source, key, week, wetday_threshold, nlag, model, hdate_last,force_download):
	# if not force_download:
	# 	try:
	# 		ff=open("modelfcst_RFREQ_"+mon+"_fday"+str(fday)+"_wk"+str(week)+".tsv", 'r')
	# 		s = ff.readline()
	# 	except OSError as err:
	# 		#print("OS error: {0}".format(err))
	# 		print("Forecasts file doesn't exist --SOLVING: downloading file")
	# 		force_download = True
	# if force_download:
	# 	#dictionary:  #CFSv2 needs to be transformed to RFREQ!
	# 	dic = {	'CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag/M%5Daverage/3./mul/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/add/4./div/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/L1/removeGRID/S/(0000%20'+str(fday)+'%20'+mon+')/VALUES/%5BS%5Daverage/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.6_hourly_rotating/.FLXF/.surface/.PRATE/%5BL%5D1/0.0/boxAverage/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag/M%5Daverage/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')VALUE/%5BX/Y%5DregridLinear/L/'+str(day1)+'/'+str(day2)+'/RANGEEDGES/%5BL%5Daverage/%5BS%5Daverage/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div/(mm/day)/unitconvert/'+str(nday)+'/mul//units/(mm)/def/exch/sub/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/1/Jan/3001/ensotime/12.0/1/Jan/3001/ensotime/%3Agrid/addGRID/T//pointwidth/0/def/pop//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
	# 			'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/'+str(wetday_threshold)+'/flagge/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1-1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/'+str(wetday_threshold)+'/flagge/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/%5Bhdate%5Daverage/sub/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/1/Jan/3001/ensotime/12.0/1/Jan/3001/ensotime/%3Agrid/addGRID/T//pointwidth/0/def/pop//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz'
	# 		  }
	# 	# calls curl to download data
	# 	url=dic[model]
	# 	print("\n Forecast URL: \n\n "+url)
	# 	get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > modelfcst_RFREQ_"+mon+"_fday"+str(fday)+"_wk"+str(week)+".tsv.gz")
	# 	get_ipython().system("gunzip -f modelfcst_RFREQ_"+mon+"_fday"+str(fday)+"_wk"+str(week)+".tsv.gz")
	# 	#curl -g -k -b '__dlauth_id='$key'' ''$url'' > modelfcst_precip_fday${fday}.tsv
	#
	# #False force_download
	# force_download = False

	#We're using model's rainfall as predictor.
	if not force_download:
		try:
			ff=open("modelfcst_precip_"+mon+"_fday"+str(fday)+"_wk"+str(week)+".tsv", 'r')
			s = ff.readline()
		except OSError as err:
			print("\033[1mWarning:\033[0;0m {0}".format(err))
			print("Forecast file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
			force_download = True
	if force_download:
		#dictionary:
		dic = {	'CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag/M%5Daverage/3./mul/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/add/4./div/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/L1/removeGRID/S/(0000%20'+str(fday)+'%20'+mon+')/VALUES/%5BS%5Daverage/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.6_hourly_rotating/.FLXF/.surface/.PRATE/%5BL%5D1/0.0/boxAverage/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag/M%5Daverage/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')VALUE/%5BX/Y%5DregridLinear/L/'+str(day1)+'/'+str(day2)+'/RANGEEDGES/%5BL%5Daverage/%5BS%5Daverage/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div/(mm/day)/unitconvert/'+str(nday)+'/mul//units/(mm)/def/exch/sub/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/1/Jan/3001/ensotime/12.0/1/Jan/3001/ensotime/%3Agrid/addGRID/T//pointwidth/0/def/pop//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
				'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/%5BM%5Daverage/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/%5BM%5Daverage/%5Bhdate%5Daverage/sub/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/1/Jan/3001/ensotime/12.0/1/Jan/3001/ensotime/%3Agrid/addGRID/T//pointwidth/0/def/pop//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
				'ECMWFrt': 'http://iridl.ldeo.columbia.edu/home/.jingyuan/.ECMWF/.realtime_S2S/.ECMF/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(last)/VALUE/%5BL%5Ddifferences/%5BM%5Daverage/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/%5BM%5Daverage/%5Bhdate%5Daverage/sub/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/1/Jan/3001/ensotime/12.0/1/Jan/3001/ensotime/%3Agrid/addGRID/T//pointwidth/0/def/pop//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
				'GEFS':           'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.EMC/.GEFS/.forecast/.pr/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUES/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/RANGEEDGES/%5BM%5Daverage/L/'+str(nday)+'/runningAverage/SOURCES/.Models/.SubX/.EMC/.GEFS/.hindcast/.dc9915/.pr/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/RANGEEDGES/L/'+str(nday)+'/runningAverage/S/(T)/renameGRID/pentadmean/T/(S)/renameGRID/%5BS%5DregridLinear/S/1/setgridtype/pop/S/2/index/.S/SAMPLE/sub/c%3A/0.001/(m3%20kg-1)/%3Ac/mul/c%3A/1000/(mm%20m-1)/%3Ac/mul/c%3A/86400/(s%20day-1)/%3Ac/mul/c%3A/7.0//units//days/def/%3Ac/mul/S/(T)/renameGRID/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/3001/ensotime/12.0/16/Jan/3001/ensotime/%3Agrid/use_as_grid/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz'
			  }
		# calls curl to download data
		url=dic[model]
		print("\n Forecast URL: \n\n "+url)
		get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > modelfcst_precip_"+mon+"_fday"+str(fday)+"_wk"+str(week)+".tsv.gz")
		get_ipython().system("gunzip -f modelfcst_precip_"+mon+"_fday"+str(fday)+"_wk"+str(week)+".tsv.gz")
		#curl -g -k -b '__dlauth_id='$key'' ''$url'' > modelfcst_precip_fday${fday}.tsv

	#False force_download
	force_download = False

	#The next two if-blocks are used for noMOS forecasts ##Added by AGM
	#Short hindcast to correctly compute climatological period of the forecast
	if not force_download:
		try:
			ff=open("noMOS/modelshort_precip_"+mon+"_wk"+str(week)+".tsv", 'r')
			s = ff.readline()
		except OSError as err:
			print("\033[1mWarning:\033[0;0m {0}".format(err))
			print("Short hindcast file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
			force_download = True
	if force_download:
		#dictionary:
		dic = { 'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(fyr-1)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/T/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1920/ensotime/%3Agrid/replaceGRID//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
		'ECMWFrt': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(last)/VALUE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(fyr-1)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/T/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1920/ensotime/%3Agrid/replaceGRID//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
		}
		# calls curl to download data
		url=dic[model]
		print("\n Short hindcast file\n") #URL: \n\n "+url)
		get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > noMOS/modelshort_precip_"+mon+"_wk"+str(week)+".tsv.gz")
		get_ipython().system("gunzip -f noMOS/modelshort_precip_"+mon+"_wk"+str(week)+".tsv.gz")

	#Short obs period corresponding to the short hindcast period
	#False force_download
	force_download = False

	if not force_download:
		try:
			ff=open("noMOS/obsshort_RFREQ_"+mon+"_wk"+str(week)+".tsv", 'r')
			s = ff.readline()
		except OSError as err:
			print("\033[1mWarning:\033[0;0m {0}".format(err))
			print("Short obs precip file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
			force_download = True
	#dictionaries:
	if threshold_pctle:
		dic = { 'CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/'+str(hstep)+'/STEP/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/T/(1%20Jan%201999)/(31%20Dec%202011)/RANGEEDGES/%5BT%5Dpercentileover/'+str(wetday_threshold)+'/flagle/T/'+str(nday)+'/runningAverage/'+str(nday)+'/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/%3Agrid/use_as_grid//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
				'ECMWF':'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(fyr-1)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/%5BT%5Dpercentileover/'+str(wetday_threshold)+'/flagle/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1920/ensotime/%3Agrid/use_as_grid//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
				'ECMWFrt':'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(last)/VALUE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(fyr-1)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/%5BT%5Dpercentileover/'+str(wetday_threshold)+'/flagle/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1920/ensotime/%3Agrid/use_as_grid//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz'
				}
	else:
		dic = { 'CFSv2':                     'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/'+str(hstep)+'/STEP/L1/S/add/0/RECHUNK/name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/'+str(wetday_threshold)+'/flagge/dup/pentadmean/%5BT%5D/regridLinear/sub/T/'+str(nday)+'/runningAverage/c%3A/7.0//units//days/def/%3Ac/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/I/3/-1/roll/.T/replaceGRID/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/%3Agrid/use_as_grid/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
				'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(fyr-1)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/'+str(wetday_threshold)+'/flagge/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1920/ensotime/%3Agrid/use_as_grid//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
				'ECMWFrt': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(last)/VALUE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(fyr-1)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/'+str(wetday_threshold)+'/flagge/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1920/ensotime/%3Agrid/use_as_grid//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
				'GEFS':       'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.EMC/.GEFS/.hindcast/.pr/S/(0000%206%20Jan%201999)/(0000%2028%20Dec%202015)/RANGEEDGES/L/('+str(day1)+')/('+str(day2)+')/RANGEEDGES/L/'+str(nday)+'/runningAverage/S/('+training_season+')/VALUES/L/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.NASA/.GES-DAAC/.TRMM_L3/.TRMM_3B42/.v7/.daily/.precipitation/X/0./1.5/360./GRID/Y/-50/1.5/50/GRID/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/'+str(wetday_threshold)+'/flagge/dup/pentadmean/%5BT%5D/regridLinear/sub/T/'+str(nday)+'/runningAverage/c%3A/7.0//units//days/def/%3Ac/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/I/3/-1/roll/.T/replaceGRID/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2301/ensotime/%3Agrid/use_as_grid/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz'
				}
		# calls curl to download data
		url=dic[model]
		print("\n Short hindcast file\n") #URL: \n\n "+url)
		get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > noMOS/modelshort_precip_"+mon+"_wk"+str(week)+".tsv.gz")
		get_ipython().system("gunzip -f noMOS/obsshort_RFREQ_"+mon+"_wk"+str(week)+".tsv.gz")

	#False force_download
	force_download = False

	#The next block is used for noMOS probabilistic forecasts ##Added by AGM
	#Above normal:
	if not force_download:
		try:
			ff=Dataset('noMOS/modelfcst_above_RFREQ_'+mon+'_wk'+str(week)+'.nc', 'r')
			s = ff.variables['Y'][:]
		except OSError as err:
			print("\033[1mWarning:\033[0;0m {0}".format(err))
			print("Above normal probability forecast file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
			force_download = True
	if force_download:
		#dictionary:
		dic = { 'ECMWF': 'http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/hdate/('+str(fyr-20)+')/('+str(hdate_last)+')/RANGE%5BL%5Ddifferences%5BM%5Daverage%5Bhdate%5D0.33/0.66/0/replacebypercentile/percentile/0.66/VALUE/flaggt%5BM%5Daverage/100/mul//long_name/%28Probability%20of%20Above%20Normal%20Tercile%29def//units/%28%25%29def/data.nc',
		'ECMWFrt': 'http://iridl.ldeo.columbia.edu/home/.jingyuan/.ECMWF/.realtime_S2S/.ECMF/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(last)/VALUE/%5BL%5Ddifferences/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/hdate/('+str(fyr-20)+')/('+str(hdate_last)+')/RANGE%5BL%5Ddifferences%5BM%5Daverage%5Bhdate%5D0.33/0.66/0/replacebypercentile/percentile/0.66/VALUE/flaggt%5BM%5Daverage/100/mul//long_name/%28Probability%20of%20Above%20Normal%20Tercile%29def//units/%28%25%29def/data.nc',
		}
		# calls curl to download data
		url=dic[model]
		#print("\n Short hindcast URL: \n\n "+url)
		get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > noMOS/modelfcst_above_RFREQ_"+mon+"_wk"+str(week)+".nc")

	#False force_download
	force_download = False

	#Below normal:
	if not force_download:
		try:
			ff=Dataset("noMOS/modelfcst_below_RFREQ_"+mon+"_wk"+str(week)+".nc", 'r')
			s = ff.variables['Y'][:]
		except OSError as err:
			print("\033[1mWarning:\033[0;0m {0}".format(err))
			print("Below normal probability forecast file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
			force_download = True
	if force_download:
		#dictionary:
		dic = { 'ECMWF': 'http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/hdate/('+str(fyr-20)+')/('+str(hdate_last)+')/RANGE%5BL%5Ddifferences%5BM%5Daverage%5Bhdate%5D0.33/0.66/0/replacebypercentile/percentile/0.33/VALUE/flaglt%5BM%5Daverage/100/mul//long_name/%28Probability%20of%20Below%20Normal%20Tercile%29def//units/%28%25%29def/data.nc',
		'ECMWFrt': 'http://iridl.ldeo.columbia.edu/home/.jingyuan/.ECMWF/.realtime_S2S/.ECMF/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(last)/VALUE/%5BL%5Ddifferences/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/hdate/('+str(fyr-20)+')/('+str(hdate_last)+')/RANGE%5BL%5Ddifferences%5BM%5Daverage%5Bhdate%5D0.33/0.66/0/replacebypercentile/percentile/0.33/VALUE/flaglt%5BM%5Daverage/100/mul//long_name/%28Probability%20of%20Below%20Normal%20Tercile%29def//units/%28%25%29def/data.nc',
		}
		# calls curl to download data
		url=dic[model]
		#print("\n Short hindcast URL: \n\n "+url)
		get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > noMOS/modelfcst_below_RFREQ_"+mon+"_wk"+str(week)+".nc")

	#False force_download
	force_download = False

	#The next block is used for noMOS flexible probabilistic forecasts ##Added by AGM
	#Ensemble mean:
	if not force_download:
		try:
			ff=Dataset('noMOS/modelfcst_mu_PRCP_'+mon+'_wk'+str(week)+'.nc', 'r')
			s = ff.variables['Y'][:]
		except OSError as err:
			print("\033[1mWarning:\033[0;0m {0}".format(err))
			print("Ensemble mean file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
			force_download = True
	if force_download:
		#dictionary:
		dic = { 'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/%5BM%5Daverage/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/%5BM%5Daverage/%5Bhdate%5Daverage/sub/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/data.nc',
		'ECMWFrt': 'http://iridl.ldeo.columbia.edu/home/.jingyuan/.ECMWF/.realtime_S2S/.ECMF/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(last)/VALUE/%5BL%5Ddifferences/%5BM%5Daverage/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/%5BM%5Daverage/%5Bhdate%5Daverage/sub/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/data.nc',
		}
		# calls curl to download data
		url=dic[model]
		print("\n Ensemble mean URL: \n\n "+url)
		get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > noMOS/modelfcst_mu_PRCP_"+mon+"_wk"+str(week)+".nc")

	#False force_download
	force_download = False

	#Ensemble standard deviation:
	if not force_download:
		try:
			ff=Dataset("noMOS/modelfcst_std_PRCP_"+mon+"_wk"+str(week)+".nc", 'r')
			s = ff.variables['Y'][:]
		except OSError as err:
			print("\033[1mWarning:\033[0;0m {0}".format(err))
			print("Ensemble standard deviation file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
			force_download = True
	if force_download:
		#dictionary:
		dic = { 'ECMWF': 'http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/%5BM%5Drmsover/c://name//water_density/def/998/%28kg/m3%29/:c/div//mm/unitconvert/data.nc',
				'ECMWFrt': 'http://iridl.ldeo.columbia.edu/home/.jingyuan/.ECMWF/.realtime_S2S/.ECMF/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(last)/VALUE/%5BL%5Ddifferences/%5BM%5Drmsover/c://name//water_density/def/998/%28kg/m3%29/:c/div//mm/unitconvert/data.nc',
				}
		# calls curl to download data
		url=dic[model]
		#print("\n Ensemble std URL: \n\n "+url)
		get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > noMOS/modelfcst_std_PRCP_"+mon+"_wk"+str(week)+".nc")

	#Obs mean:
	if not force_download:
		try:
			ff=Dataset('noMOS/obs_mu_PRCP_'+mon+'_wk'+str(week)+'.nc', 'r')
			s = ff.variables['Y'][:]
		except OSError as err:
			print("\033[1mWarning:\033[0;0m {0}".format(err))
			print("Obs mean file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
			force_download = True
	if force_download:
		#dictionary:
		dic = { 'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(hdate_last)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/%5BX/Y%5D/regridAverage/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/dup%5BT%5Daverage/sub/-999/setmissing_value/nip/T/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1920/ensotime/%3Agrid/replaceGRID//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/%5BT%5Daverage/data.nc',
		'ECMWFrt': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(last)/VALUE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(hdate_last)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/%5BX/Y%5D/regridAverage/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/dup%5BT%5Daverage/sub/-999/setmissing_value/nip/T/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1920/ensotime/%3Agrid/replaceGRID//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/%5BT%5Daverage/data.nc',
		}
		# calls curl to download data
		url=dic[model]
		#print("\n Obs mean URL: \n\n "+url)
		get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > noMOS/obs_mu_PRCP_"+mon+"_wk"+str(week)+".nc")

	#False force_download
	force_download = False

	#Obs std:
	if not force_download:
		try:
			ff=Dataset('noMOS/obs_std_PRCP_'+mon+'_wk'+str(week)+'.nc', 'r')
			s = ff.variables['Y'][:]
		except OSError as err:
			print("\033[1mWarning:\033[0;0m {0}".format(err))
			print("Obs std file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
			force_download = True
	if force_download:
		#dictionary: http://iridl.ldeo.columbia.edu/home/.jingyuan/.ECMWF/.realtime_S2S/.ECMF/
		dic = { 'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(hdate_last)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/%5BX/Y%5D/regridAverage/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/-999/setmissing_value/nip/T/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1920/ensotime/%3Agrid/replaceGRID//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/%5BT%5Drmsover/data.nc',
		'ECMWFrt': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1)+')/('+str(day2)+')/VALUES/S/(last)/VALUE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(hdate_last)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/'+obs_source+'/%5BX/Y%5D/regridAverage/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/-999/setmissing_value/nip/T/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1920/ensotime/%3Agrid/replaceGRID//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/%5BT%5Drmsover/data.nc',
		}
		# calls curl to download data
		url=dic[model]
		#print("\n Obs std URL: \n\n "+url)
		get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > noMOS/obs_std_PRCP_"+mon+"_wk"+str(week)+".nc")

def GetForecast_Temp(day1, day2, fday, mon, fyr, nday, wlo1, elo1, sla1, nla1, wlo2, elo2, sla2, nla2, obs_source, key, week, nlag, model, hdate_last, threshold_pctle,training_season,wetday_threshold,force_download):
	if not force_download:
		try:
			ff=open("modelfcst_temp_"+mon+"_fday"+str(fday)+"_wk"+str(week)+".tsv", 'r')
			s = ff.readline()
		except OSError as err:
			print("\033[1mWarning:\033[0;0m {0}".format(err))
			print("Forecast file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m")
			force_download = True
	if force_download:
		#dictionary:
		dic = {	'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.forecast/.perturbed/.sfc_temperature/.skt/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/LA/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Daverage/%5BM%5Daverage/SOURCES/.ECMWF/.S2S/.ECMF/.forecast/.perturbed/.sfc_temperature/.skt/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/LA/('+str(day1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Daverage/%5BM%5Daverage/%5Bhdate%5Daverage/sub//Celsius/unitconvert/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/1/Jan/3001/ensotime/12.0/1/Jan/3001/ensotime/%3Agrid/addGRID/T//pointwidth/0/def/pop//name/(temp)/def//units/(Celsius)/def//long_name/(surface temperature)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
			  }
		# calls curl to download data
		url=dic[model]
		print("\n Forecast URL: \n\n "+url)
		get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > modelfcst_temp_"+mon+"_fday"+str(fday)+"_wk"+str(week)+".tsv.gz")
		get_ipython().system("gunzip -f modelfcst_temp_"+mon+"_fday"+str(fday)+"_wk"+str(week)+".tsv.gz")
		#curl -g -k -b '__dlauth_id='$key'' ''$url'' > modelfcst_precip_fday${fday}.tsv

def CPTscript(mon,fday,lit,liti,wk,nla1,sla1,wlo1,elo1,nla2,sla2,wlo2,elo2,fprefix,mpref,training_season,ntrain,rainfall_frequency,MOS):
		"""Function to write CPT namelist file

		"""
		# Set up CPT parameter file
		f=open("params","w")
		if MOS=='CCA':
			# Opens CCA
			f.write("611\n")
		elif MOS=='PCR':
			# Opens PCR
			f.write("612\n")
		elif MOS=='ELR':
			# Opens GCM; because the calibration takes place via sklearn.linear_model (in the Jupyter notebook)
			f.write("614\n")
		elif MOS=='None':
			# Opens GCM (no calibration performed in CPT)
			f.write("614\n")
		else:
			print ("MOS option is invalid")

		# First, ask CPT to stop if error is encountered
		f.write("571\n")
		f.write("3\n")
		# Second, ask CPT to not show those menus again....  (deactivate this if debugging!)
		f.write("572\n")

		# Opens X input file
		f.write("1\n")
		if rainfall_frequency:
			file='../input/model_precip_'+mon+'_wk'+str(wk)+'.tsv\n'  #in the future: use model freq
		else:
			file='../input/model_precip_'+mon+'_wk'+str(wk)+'.tsv\n'
		f.write(file)
		# Nothernmost latitude
		f.write(str(nla1)+'\n')
		# Southernmost latitude
		f.write(str(sla1)+'\n')
		# Westernmost longitude
		f.write(str(wlo1)+'\n')
		# Easternmost longitude
		f.write(str(elo1)+'\n')
		if MOS=='CCA' or MOS=='PCR':
			# Minimum number of X modes
			f.write("1\n")
			# Maximum number of X modes
			f.write("10\n")

			# Opens forecast (X) file
			f.write("3\n")
			if rainfall_frequency:
				file='../input/modelfcst_precip_'+mon+'_fday'+str(fday)+'_wk'+str(wk)+'.tsv\n'
			else:
				file='../input/modelfcst_precip_'+mon+'_fday'+str(fday)+'_wk'+str(wk)+'.tsv\n'
			f.write(file)

		# Opens Y input file
		f.write("2\n")
		if rainfall_frequency:
			file='../input/obs_RFREQ_'+mon+'_wk'+str(wk)+'.tsv\n'
		else:
			file='../input/obs_precip_'+mon+'_wk'+str(wk)+'.tsv\n'
		f.write(file)
		# Nothernmost latitude
		f.write(str(nla2)+'\n')
		# Southernmost latitude
		f.write(str(sla2)+'\n')
		# Westernmost longitude
		f.write(str(wlo2)+'\n')
		# Easternmost longitude
		f.write(str(elo2)+'\n')
		if MOS=='CCA':
			# Minimum number of Y modes
			f.write("1\n")
			# Maximum number of Y modes
			f.write("10\n")

			# Minimum number of CCA modes
			f.write("1\n")
			# Maximum number of CCAmodes
			f.write("5\n")

		# X training period
		f.write("4\n")
		# First year of X training period
		f.write("1901\n")
		# Y training period
		f.write("5\n")
		# First year of Y training period
		f.write("1901\n")

		# Goodness index
		f.write("531\n")
		# Kendall's tau
		f.write("3\n")

		# Option: Length of training period
		f.write("7\n")
		# Length of training period
		f.write(str(ntrain)+'\n')
		# Option: Length of cross-validation window
		f.write("8\n")
		# Enter length
		f.write("5\n")

		# Turn ON Transform predictand data
		f.write("541\n")
		# Turn ON zero bound for Y data	 (automatically on by CPT if variable is precip)
		#f.write("542\n")
		# Turn ON synchronous predictors
		f.write("545\n")
		# Turn ON p-values for masking maps
		#f.write("561\n")

		### Missing value options
		f.write("544\n")
		# Missing value X flag:
		blurb='-999\n'
		f.write(blurb)
		# Maximum % of missing values
		f.write("10\n")
		# Maximum % of missing gridpoints
		f.write("10\n")
		# Number of near-neighbors
		f.write("1\n")
		# Missing value replacement : best-near-neighbors
		f.write("4\n")
		# Y missing value flag
		blurb='-999\n'
		f.write(blurb)
		# Maximum % of missing values
		f.write("10\n")
		# Maximum % of missing stations
		f.write("10\n")
		# Number of near-neighbors
		f.write("1\n")
		# Best near neighbor
		f.write("4\n")

		# Transformation settings
		#f.write("554\n")
		# Empirical distribution
		#f.write("1\n")

		#######BUILD MODEL AND VALIDATE IT	!!!!!

		# NB: Default output format is GrADS format
		# select output format
		f.write("131\n")
		# GrADS format
		f.write("3\n")

		# save goodness index
		f.write("112\n")
		file='../output/'+fprefix+'_'+mpref+'_Kendallstau_'+training_season+'_wk'+str(wk)+'\n'
		f.write(file)

		## Cross-validation
		#f.write("311\n")   #--deactivated

		#Retroactive for s2s, due to the large sample size - deactivated for CFSv2
		f.write("312\n")
		#Length of initial training period: (Just quits w/o error msg if 80>ntrain)
		f.write(str(lit)+'\n')
		#Update interval:
		f.write(str(liti)+'\n')   #--old comment from AGM: 80 for speeding up tests, change to 20 later (~same results so far with 20 or 80)

		if MOS=='None': #for some weird reason for None we need to run it twice for it to work (at least in v16.2*)
			# Retroactive for s2s, due to the large sample size
			f.write("312\n")
			#Length of initial training period:
			f.write(str(lit)+'\n')
			#Update interval:
			f.write(str(lit)+'\n')   #--80 for speeding up tests, change to 20 later (~same results so far with 20 or 80)

		# cross-validated skill maps
		f.write("413\n")
		# save Pearson's Correlation
		f.write("1\n")
		file='../output/'+fprefix+'_'+mpref+'_Pearson_'+training_season+'_wk'+str(wk)+'\n'
		f.write(file)

		# cross-validated skill maps
		f.write("413\n")
		# save Spearmans Correlation
		f.write("2\n")
		file='../output/'+fprefix+'_'+mpref+'_Spearman_'+training_season+'_wk'+str(wk)+'\n'
		f.write(file)

		# cross-validated skill maps
		f.write("413\n")
		# save 2AFC score
		f.write("3\n")
		file='../output/'+fprefix+'_'+mpref+'_2AFC_'+training_season+'_wk'+str(wk)+'\n'
		f.write(file)

		# cross-validated skill maps
		f.write("413\n")
		# save RocBelow score
		f.write("15\n")
		file='../output/'+fprefix+'_'+mpref+'_RocBelow_'+training_season+'_wk'+str(wk)+'\n'
		f.write(file)

		# cross-validated skill maps
		f.write("413\n")
		# save RocAbove score
		f.write("16\n")
		file='../output/'+fprefix+'_'+mpref+'_RocAbove_'+training_season+'_wk'+str(wk)+'\n'
		f.write(file)

		#Now implementing forecasts for also noMOS case. Perhaps the best is to compute everything in the DL.
		#if MOS=='CCA' or MOS=='PCR':   #DO NOT USE CPT to compute probabilities if MOS='None' --use IRIDL for direct counting

		if MOS=='None':
		#######Probabilistic Forecasts Verification for NoMOS (PFV) --already computed if using retroactive option
			#Reliability diagram
			f.write("431\n")
			f.write("Y\n") #yes, save results to a file
			file='../output/'+fprefix+'_'+mpref+'RFCST_reliabdiag_'+training_season+'_'+mon+str(fday)+'_wk'+str(wk)+'.txt\n'
			f.write(file)

			# select output format -- GrADS, so we can plot it in Python
			f.write("131\n")
			# GrADS format
			f.write("3\n")

			# Probabilistic skill maps
			f.write("437\n")
			# save Ignorance (all cats)
			f.write("101\n")
			file='../output/'+fprefix+'_'+mpref+'_Ignorance_'+training_season+'_wk'+str(wk)+'\n'
			f.write(file)

			# Probabilistic skill maps
			f.write("437\n")
			# save Ranked Probability Skill Score (all cats)
			f.write("122\n")
			file='../output/'+fprefix+'_'+mpref+'_RPSS_'+training_season+'_wk'+str(wk)+'\n'
			f.write(file)

			# Probabilistic skill maps
			f.write("437\n")
			# save Ranked Probability Skill Score (all cats)
			f.write("131\n")
			file='../output/'+fprefix+'_'+mpref+'_GROC_'+training_season+'_wk'+str(wk)+'\n'
			f.write(file)

		#######FORECAST(S)	!!!!!
			# Re-opens X input file and use the short hindcasts so climo is consistent with forecast file
			f.write("1\n")
			f.write("Y\n")  #Yes to cleaning current results
			if rainfall_frequency:
				file='../input/noMOS/modelshort_precip_'+mon+'_wk'+str(wk)+'.tsv\n'  #in the future: use model freq
			else:
				file='../input/noMOS/modelshort_precip_'+mon+'_wk'+str(wk)+'.tsv\n'
			f.write(file)
			# Nothernmost latitude
			f.write(str(nla1)+'\n')
			# Southernmost latitude
			f.write(str(sla1)+'\n')
			# Westernmost longitude
			f.write(str(wlo1)+'\n')
			# Easternmost longitude
			f.write(str(elo1)+'\n')

			# Just in case CPT is confused: Open forecast (X) file
			f.write("3\n")
			if rainfall_frequency:
				file='../input/modelfcst_precip_'+mon+'_fday'+str(fday)+'_wk'+str(wk)+'.tsv\n'
			else:
				file='../input/modelfcst_precip_'+mon+'_fday'+str(fday)+'_wk'+str(wk)+'.tsv\n'
			f.write(file)

			# Re-opens Y input file, and use short version to be consistent with hindcasts above
			f.write("2\n")
			if rainfall_frequency:
				file='../input/noMOS/obsshort_RFREQ_'+mon+'_wk'+str(wk)+'.tsv\n'
			else:
				file='../input/noMOS/obsshort_precip_'+mon+'_wk'+str(wk)+'.tsv\n'
			f.write(file)
			# Nothernmost latitude
			f.write(str(nla2)+'\n')
			# Southernmost latitude
			f.write(str(sla2)+'\n')
			# Westernmost longitude
			f.write(str(wlo2)+'\n')
			# Easternmost longitude
			f.write(str(elo2)+'\n')

			# Need to shorten the length of training period
			f.write("7\n")
			# Length of training period
			f.write("20\n")

			# Cross-validation due to the shorter sample (only 20 steps) --just for forecast purposes. Skill computed with retroactive
			f.write("311\n")

		# Probabilistic (3 categories) maps
		f.write("455\n")
		# Output results
		f.write("111\n")
		# Forecast probabilities
		f.write("501\n")
		file='../output/'+fprefix+'_'+mpref+'FCST_P_'+training_season+'_'+mon+str(fday)+'_wk'+str(wk)+'\n'
		f.write(file)
		#502 # Forecast odds
		#Exit submenu
		f.write("0\n")

		# Compute deterministic values and prediction limits
		f.write("454\n")
		# Output results
		f.write("111\n")
		# Forecast values
		f.write("511\n")
		file='../output/'+fprefix+'_'+mpref+'FCST_V_'+training_season+'_'+mon+str(fday)+'_wk'+str(wk)+'\n'
		f.write(file)
		#502 # Forecast odds


		#######Following files are used to plot the flexible format
		# Save cross-validated predictions
		f.write("201\n")
		file='../output/'+fprefix+'_'+mpref+'FCST_xvPr_'+training_season+'_'+mon+str(fday)+'_wk'+str(wk)+'\n'
		f.write(file)
		# Save deterministic forecasts [mu for Gaussian fcst pdf]
		f.write("511\n")
		file='../output/'+fprefix+'_'+mpref+'FCST_mu_'+training_season+'_'+mon+str(fday)+'_wk'+str(wk)+'\n'
		f.write(file)
		# Save prediction error variance [sigma^2 for Gaussian fcst pdf]
		f.write("514\n")
		file='../output/'+fprefix+'_'+mpref+'FCST_var_'+training_season+'_'+mon+str(fday)+'_wk'+str(wk)+'\n'
		f.write(file)
		# Save z
		#f.write("532\n")
		#file='../output/'+fprefix+'_'+mpref+'FCST_z_'+training_season+'_'+mon+str(fday)+'_wk'+str(wk)+'\n'
		#f.write(file)
		# Save predictand [to build predictand pdf]
		f.write("102\n")
		file='../output/'+fprefix+'_'+mpref+'FCST_Obs_'+training_season+'_'+mon+str(fday)+'_wk'+str(wk)+'\n'
		f.write(file)

		#Exit submenu
		f.write("0\n")

		if MOS=='CCA' or MOS=='PCR':
			# ###########PFV --Added by AGM in version 1.5
			# #Compute and write retrospective forecasts for prob skill assessment.
			# #Re-define forecas file
			# f.write("3\n")
			# if rainfall_frequency:
			# 	file='../input/model_precip_'+mon+'_wk'+str(wk)+'.tsv\n'  #in the future: use model freq
			# else:
			# 	file='../input/model_precip_'+mon+'_wk'+str(wk)+'.tsv\n'
			# f.write(file)
			# #Forecast period settings
			# f.write("6\n")
			# # First year to forecast. Save ALL forecasts (for "retroactive" we should only assess second half)
			# f.write("1901\n")
			# #Number of forecasts option
			# f.write("9\n")
			# # Number of reforecasts to produce
			# f.write("160\n")
			# # Change to ASCII format to re0use in CPT
			# f.write("131\n")
			# # ASCII format
			# f.write("2\n")
			# # Probabilistic (3 categories) maps
			# f.write("455\n")
			# # Output results
			# f.write("111\n")
			# # Forecast probabilities --Note change in name for reforecasts:
			# f.write("501\n")
			# file='../output/'+fprefix+'_'+mpref+'RFCST_P_'+training_season+'_'+mon+str(fday)+'_wk'+str(wk)+'\n'
			# f.write(file)
			# #502 # Forecast odds
			# #Exit submenu
			# f.write("0\n")
			#
			# # Close X file so we can access the PFV option
			# f.write("121\n")
			# f.write("Y\n")  #Yes to cleaning current results:# WARNING:
			# #Select Probabilistic Forecast Verification (PFV)
			# f.write("621\n")
			# # Opens X input file
			# f.write("1\n")
			# file='../output/'+fprefix+'_'+mpref+'RFCST_P_'+training_season+'_'+mon+str(fday)+'_wk'+str(wk)+'.txt\n'
			# f.write(file)
			# # Nothernmost latitude
			# f.write(str(nla2)+'\n')
			# # Southernmost latitude
			# f.write(str(sla2)+'\n')
			# # Westernmost longitude
			# f.write(str(wlo2)+'\n')
			# # Easternmost longitude
			# f.write(str(elo2)+'\n')
			#
			# f.write("5\n")
			# # First year of the PFV
			# # for "retroactive" only second half of the entire period should be used --this value is for ECMWF only)
			# fypfv=1901+lit
			# f.write(str(fypfv)+'\n')
			# #f.write("1901\n")
			#
			# #Verify
			# f.write("313\n")
			#
			# #Reliability diagram
			# f.write("431\n")
			# f.write("Y\n") #yes, save results to a file
			# file='../output/'+fprefix+'_'+mpref+'RFCST_reliabdiag_'+training_season+'_'+mon+str(fday)+'_wk'+str(wk)+'.txt\n'
			# f.write(file)
			#
			# # select output format -- GrADS, so we can plot it in Python
			# f.write("131\n")
			# # GrADS format
			# f.write("3\n")

			# Probabilistic skill maps
			f.write("437\n")
			# save Ignorance (all cats)
			f.write("101\n")
			file='../output/'+fprefix+'_'+mpref+'_Ignorance_'+training_season+'_wk'+str(wk)+'\n'
			f.write(file)

			# Probabilistic skill maps
			f.write("437\n")
			# save Ranked Probability Skill Score (all cats)
			f.write("122\n")
			file='../output/'+fprefix+'_'+mpref+'_RPSS_'+training_season+'_wk'+str(wk)+'\n'
			f.write(file)

			# Probabilistic skill maps
			f.write("437\n")
			# save Ranked Probability Skill Score (all cats)
			f.write("131\n")
			file='../output/'+fprefix+'_'+mpref+'_GROC_'+training_season+'_wk'+str(wk)+'\n'
			f.write(file)

		# Exit
		f.write("0\n")

		f.close()
		get_ipython().system('cp params '+fprefix+'_'+mpref+'_'+training_season+'_'+mon+str(fday)+'_wk'+str(wk)+'.cpt')
