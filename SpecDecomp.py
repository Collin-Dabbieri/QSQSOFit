# CMD 5/17/22 I'm changing the fit emcee function so that the user is always providing the same dictionary for params.
# emcee needs a params list, so the function will take a dictionary from the user and convert it to the list
# I'm also changing the way narrow emission line params are linked. Instead of a constant offset between the wavelength of the emission lines
# They will be linked according to the wavelength ratios. Essentially you have delta lambda / lambda = v / c
# and you want different narrow lines to have the same v. Since different lines have different rest wavelengths, you want to tie them via wavelength ratios
# I'm also adding the ability to shift the templates left and right
# so the template fitter will have three params: index, scale factor, and shift
# remember index is width for Fe II, age of stellar population for host galaxy, etc.
# I want to add an ability for the user to select which model params to use
# For example, you could decide to turn off He II or the Balmer continuum
# And I want the user to be able to decide how many gaussians to use for the broad and narrow lines

# TODO create an option for the user to be able to specify which gaussians they want and how they are tied, fixed, etc. 
# pyspeckit does this in a good way. It might be easiest to just use pyspeckit inside this code

import math
import numpy as np
from matplotlib import pyplot as plt
import os
import glob
from lmfit import Parameters, fit_report, minimize
import time
import emcee

# params['pl_norm']
# params['pl_alpha']
# params['fe_ii_width']
# params['fe_ii_scale']
# params['host_age']
# params['host_scale']
# params['bal_cont_tau']
# params['bal_cont_scale']
# params['bal_high_width']
# params['bal_high_scale']
#params['ampOII3729']
#params['ampOII3735']
#params['ampNeIII3868']
#params['ampNeIII3966']
#params['ampSII4073']
#params['groupampratio']
#params['groupWLoffset1']
#params['groupWLoffset2']
#params['groupwidth1']
#params['groupwidth2']
#params['ampHDelta']
#params['WLHDelta']
#params['widthHDelta']
#params['ampHGamma']
#params['WLHGamma']
#params['widthHGamma']
#params['ampHeII']
#params['WLHeII']
#params['widthHeII']
#params['ampHBeta1']
#params['WLHBeta1']
#params['widthHBeta1']
#params['ampHBeta2']
#params['WLHBeta2']
#params['widthHBeta2']
#params['ampHBeta3']
#params['WLHBeta3']
#params['widthHBeta3']
#params['ampHBeta4']
#params['WLHBeta4']
#params['widthHBeta4']
#params['ampNHBeta']
#params['ampOIII5007']
#params['group2ampratio']
#params['group2WLoffset1']
#params['group2WLoffset2']
#params['group2width1']
#params['group2width2']  

####################################################################################################################################
############# SpecDecomp Class
####################################################################################################################################

class SpecDecomp():
    
    def __init__(self,wave,flux,err):
        '''
        Initializes spectrum, grabs FeII templates and stellar templates
        '''
        
        # prune bad values from wave,flux, and err
        self.wave_init=wave
        self.flux_init=flux
        self.err_init=err
        self.wave,self.flux,self.err=self.clip_data(wave,flux,err)
        
        feii_template_path='./files/fe_op_templates.npy'
        stellar_template_path='./files/PyQSOfit_MILES_templates.dat'
        balmer_cont_template_path='./files/balmer_cont_templates.npy'
        balmer_highorder_template_path='./files/balmer_highorder_templates.npy'
        
        self.feii_templates=np.load(feii_template_path,allow_pickle=True)
        self.stellar_templates=np.genfromtxt(stellar_template_path,skip_header=5)
        self.balmer_cont_templates=np.load(balmer_cont_template_path,allow_pickle=True)
        self.balmer_highorder_templates=np.load(balmer_highorder_template_path,allow_pickle=True)
        
        # for the default state, we'll initialize with all model params
        # the user can change this by using the set_model function
        model_select={}
        model_select['PowerLaw']=True
        model_select['BalmerCont']=True
        model_select['BalmerHigh']=True
        model_select['FeII']=True
        model_select['Host']=True
        model_select['OII3729']=True
        model_select['OII3735']=True
        model_select['NeIII3868']=True
        model_select['NeIII3966']=True
        model_select['SII4073']=True
        model_select['HDelta']=True
        model_select['HGamma']=True
        model_select['HeII']=True
        model_select['BroadHbeta']=True
        model_select['NarrowHbeta']=True
        model_select['OIII4960']=True
        model_select['OIII5007']=True
        self.model_select=model_select
        
        # for the default state, we'll initialize with
        # - 2 gaussians for narrow lines
        # - 4 gaussians for broad hbeta
        # - 1 gaussian for Hdelta
        # - 1 gaussian for Hgamma
        # - 1 gaussian for He II
        # use set_num_gauss function to change
        # THIS DOES NOT WORK YET, WANT TO ADD LATER
        self.ngauss_narrow=2
        self.ngauss_bhbeta=4
        self.ngauss_hdelta=1
        self.ngauss_hgamma=1
        self.ngauss_heii=1
        
        # Initialize with some default values for lower and upper bound dictionaries
        # use set_bounds function to change
        self.OII1_WL=3729.225
        self.OII2_WL=3735.784
        self.NeIII1_WL=3868.265
        self.NeIII2_WL=3966.244
        self.SII_WL=4073.125
        self.Hdelta_WL=4102
        self.Hgamma_WL=4341
        self.HeII_WL=4685
        self.Hbeta_WL=4862.721
        self.OIII1_WL=4960.295
        self.OIII2_WL=5008.239
    
        max_broad_offset=20 #angstroms
        max_narrow_offset=10 #angstroms
        min_amp=0.1
        max_amp=100
        min_width=1        #angstroms
        min_broad_width=10 #angstroms
        broad_max=50       # max width for broad lines
        narrow_max=10      # max width for narrow lines
        
        num_templates_feii=100
        num_templates_host=8
        num_templates_balmer_cont=20
        num_templates_balmer_highorder=21

        lower_bounds={}
        lower_bounds['pl_norm']=0
        lower_bounds['pl_alpha']=-10
        lower_bounds['fe_ii_width']=0.1
        lower_bounds['fe_ii_scale']=0.1
        lower_bounds['host_age']=0.1
        lower_bounds['host_scale']=0.1
        lower_bounds['bal_cont_tau']=0.1
        lower_bounds['bal_cont_scale']=0.1
        lower_bounds['bal_high_width']=0.1
        lower_bounds['bal_high_scale']=0.1
        lower_bounds['ampOII3729']=min_amp
        lower_bounds['ampOII3735']=min_amp
        lower_bounds['ampNeIII3868']=min_amp
        lower_bounds['ampNeIII3966']=min_amp
        lower_bounds['ampSII4073']=min_amp
        lower_bounds['groupampratio']=0
        lower_bounds['groupWLoffset1']=-5 #angstroms
        lower_bounds['groupWLoffset2']=-5 #angstroms
        lower_bounds['groupwidth1']=min_width
        lower_bounds['groupwidth2']=min_width
        lower_bounds['ampHDelta']=min_amp
        lower_bounds['WLHDelta']=self.Hdelta_WL-max_broad_offset
        lower_bounds['widthHDelta']=min_broad_width
        lower_bounds['ampHGamma']=min_amp
        lower_bounds['WLHGamma']=self.Hgamma_WL-max_broad_offset
        lower_bounds['widthHGamma']=min_broad_width
        lower_bounds['ampHeII']=min_amp
        lower_bounds['WLHeII']=self.HeII_WL-max_broad_offset
        lower_bounds['widthHeII']=min_broad_width
        lower_bounds['ampHBeta1']=min_amp
        lower_bounds['WLHBeta1']=self.Hbeta_WL-max_broad_offset
        lower_bounds['widthHBeta1']=min_broad_width
        lower_bounds['ampHBeta2']=min_amp
        lower_bounds['WLHBeta2']=self.Hbeta_WL-max_broad_offset
        lower_bounds['widthHBeta2']=min_broad_width
        lower_bounds['ampHBeta3']=min_amp
        lower_bounds['WLHBeta3']=self.Hbeta_WL-max_broad_offset
        lower_bounds['widthHBeta3']=min_broad_width
        lower_bounds['ampHBeta4']=min_amp
        lower_bounds['WLHBeta4']=self.Hbeta_WL-max_broad_offset
        lower_bounds['widthHBeta4']=min_broad_width
        lower_bounds['ampNHBeta']=min_amp
        lower_bounds['ampOIII5007']=min_amp
        lower_bounds['group2ampratio']=0
        lower_bounds['group2WLoffset1']=-5 #angstroms
        lower_bounds['group2WLoffset2']=-5 #angstroms
        lower_bounds['group2width1']=min_width
        lower_bounds['group2width2']=min_width
        
        self.lower_bounds=lower_bounds
        
        upper_bounds={}
        upper_bounds['pl_norm']=150
        upper_bounds['pl_alpha']=4
        upper_bounds['fe_ii_width']=num_templates_feii-1.1
        upper_bounds['fe_ii_scale']=200
        upper_bounds['host_age']=num_templates_host-1.1
        upper_bounds['host_scale']=200
        upper_bounds['bal_cont_tau']=num_templates_balmer_cont-1.1
        upper_bounds['bal_cont_scale']=200
        upper_bounds['bal_high_width']=num_templates_balmer_highorder-1.1
        upper_bounds['bal_high_scale']=200
        upper_bounds['ampOII3729']=max_amp
        upper_bounds['ampOII3735']=max_amp
        upper_bounds['ampNeIII3868']=max_amp
        upper_bounds['ampNeIII3966']=max_amp
        upper_bounds['ampSII4073']=max_amp
        upper_bounds['groupampratio']=100
        upper_bounds['groupWLoffset1']=5 #angstroms
        upper_bounds['groupWLoffset2']=5 #angstroms
        upper_bounds['groupwidth1']=narrow_max
        upper_bounds['groupwidth2']=narrow_max
        upper_bounds['ampHDelta']=max_amp
        upper_bounds['WLHDelta']=self.Hdelta_WL+max_broad_offset
        upper_bounds['widthHDelta']=broad_max
        upper_bounds['ampHGamma']=max_amp
        upper_bounds['WLHGamma']=self.Hgamma_WL+max_broad_offset
        upper_bounds['widthHGamma']=broad_max
        upper_bounds['ampHeII']=max_amp
        upper_bounds['WLHeII']=self.HeII_WL+max_broad_offset
        upper_bounds['widthHeII']=broad_max
        upper_bounds['ampHBeta1']=max_amp
        upper_bounds['WLHBeta1']=self.Hbeta_WL+max_broad_offset
        upper_bounds['widthHBeta1']=broad_max
        upper_bounds['ampHBeta2']=max_amp
        upper_bounds['WLHBeta2']=self.Hbeta_WL+max_broad_offset
        upper_bounds['widthHBeta2']=broad_max
        upper_bounds['ampHBeta3']=max_amp
        upper_bounds['WLHBeta3']=self.Hbeta_WL+max_broad_offset
        upper_bounds['widthHBeta3']=broad_max
        upper_bounds['ampHBeta4']=max_amp
        upper_bounds['WLHBeta4']=self.Hbeta_WL+max_broad_offset
        upper_bounds['widthHBeta4']=broad_max
        upper_bounds['ampNHBeta']=max_amp
        upper_bounds['ampOIII5007']=max_amp
        upper_bounds['group2ampratio']=100
        upper_bounds['group2WLoffset1']=5 #angstroms
        upper_bounds['group2WLoffset2']=5 #angstroms
        upper_bounds['group2width1']=narrow_max
        upper_bounds['group2width2']=narrow_max
        
        self.upper_bounds=upper_bounds
        
        # list of lists, gives WL regions for initial fit w only power law and Fe II
        self.pl_feii_wls=[[4150,4250],[4450,4800],[5050,5750]]
        self.nan_policy='raise'
        # list of lists, gives WL regions for host + narrow line fits near 4000 A
        self.host_narrow_wls=[[3750,4080]]
        # list of lists, gives WL regions for H Delta and H Gamma fits
        self.hdelta_hgamma_wls=[[4000,4500]]
        # list of lists, gives WL regions for He II, H Beta and O III fits
        self.hbeta_oiii_wls=[[4600,5100]]
        
        self.pl_norm_wl=4000  # Wavelength at which the power law is normalized
        
        self.broad_start=15    # starting width for broad lines
        self.narrow_start=5    # starting width for narrow lines
          
    def clip_data(self,wave,flux,err):
        '''
        removes indexes with bad data points (err=0, nan, inf)
        '''
        
        bad_wave=np.logical_or(np.isnan(wave),np.isinf(wave))
        bad_flux=np.logical_or(np.isnan(flux),np.isinf(flux))
        bad_err=np.logical_or(np.logical_or(np.isnan(err),np.isinf(err)),np.where(err==0,True,False))
        
        bad=np.logical_or(np.logical_or(bad_wave,bad_flux),bad_err)
        good=np.logical_not(bad)
        
        return wave[good],flux[good],err[good]
    
    
    def wave_region_select(self,wave,wave_regions):
        '''
        given a wavelength vector and a list of lists with desired wavelength ranges,
        returns the indexes of wave within wave_regions
        params:
            wave - vector of wavelength values
            wave_regions - list of lists, each entry is [min_WL,max_WL] for a given desired region
        returns:
            indexes - indexes of wave within wave_regions
        '''
        indexes=[]

        for i in wave_regions:
            min_wave=i[0]
            max_wave=i[1]

            idx_iter=np.where(np.logical_and(wave>=min_wave, wave<=max_wave))[0]

            indexes.extend(idx_iter)
            
        return indexes
    
    def clip_data_wavelength(self,minWL,maxWL):
        '''
        After your class is initialized, you can use this function to clip the spectrum onto a desired wavelength range
        for fitting
        
        params:
            minWL - minimum wavelength in angstroms
            maxWL - maximum wavelength in angstroms
        '''
        
        indexes=np.where(np.logical_and(self.wave>=minWL, self.wave<=maxWL))[0]
        
        self.wave=self.wave[indexes]
        self.flux=self.flux[indexes]
        self.err=self.err[indexes]
        
    
    def set_model(self,model_select):
        '''
        Use a dictionary filled with booleans to determine which model params to use.
        Keys in the dictionary are given below, values should be True or False
        
        model_select['PowerLaw']=True
        model_select['BalmerCont']=True
        model_select['BalmerHigh']=True
        model_select['FeII']=True
        model_select['Host']=True
        model_select['OII3729']=True
        model_select['OII3735']=True
        model_select['NeIII3868']=True
        model_select['NeIII3966']=True
        model_select['SII4073']=True
        model_select['HDelta']=True
        model_select['HGamma']=True
        model_select['HeII']=True
        '''
        self.model_select=model_select
        
    def set_num_gauss(self,ngauss_narrow,ngauss_bhbeta,ngauss_hdelta,ngauss_hgamma,ngauss_heii):
        '''
        Update the number of gaussians for different emission lines
        
        THIS FUNCTIONALITY WILL COME LATER, TOO COMPLICATED FOR NOW
        '''
        self.ngauss_narrow=ngauss_narrow
        self.ngauss_bhbeta=ngauss_bhbeta
        self.ngauss_hdelta=ngauss_hdelta
        self.ngauss_hgamma=ngauss_hgamma
        self.ngauss_heii=ngauss_heii
        
    def set_bounds(self,lower_bounds,upper_bounds):
        '''
        Update lower_bounds and upper_bounds dictionaries
        '''
        
        self.lower_bounds=lower_bounds
        self.upper_bounds=upper_bounds


    def template_fitter(self,wave,templates,index,scale_factor):
        '''
        given a series of templates (like stellar models by age or Fe II templates by convolution width),
        returns a spectrum interpolated between template spectra and scaled.
        Spectrum is also interpolated onto the wavelength vector of the observed data.
        
        params:
            wave - wavelength vector of observed data
            templates - 2D numpy array of shape (numpoints,num_templates+1), where 1st column is wavelength, subsequent columns are templates
            index - selected index within templates [0-num_templates]. Float, will interpolate between templates when index!=int
            scale_factor - multiplicative scale factor for spectrum
        '''
        num_templates=templates.shape[1]-1
        numpoints=templates.shape[0]
        
        wave_temp=templates[:,0]
        flux_temp=templates[:,1:]
        
        # Perform interpolation between templates
        if index==num_templates-1:
            spectrum=templates[:,-1]
        else:
            lower_idx=int(math.floor(index)+1) #+1 because 0th index is wavelength column
            lower_spectrum=templates[:,lower_idx] # this means lower in index, not necessarily in flux
            
            upper_idx=lower_idx+1
            upper_spectrum=templates[:,upper_idx]
            
            weight=(index+1)-lower_idx #1=choose upper spectrum value, 0=choose lower spectrum value, 0.5=choose midpoint
            
            distances=upper_spectrum-lower_spectrum # array of distances between selected spectra
            
            spectrum=lower_spectrum+(weight*distances)
            
        # Perform scaling
        scaled_spectrum=spectrum*scale_factor
        
        # interpolate spectrum onto wavelength vector of data
        final_spectrum=np.interp(wave,wave_temp,scaled_spectrum)
        
        return final_spectrum
    
    def gaussian(self,wave,amp,mu,sigma):
    
        return amp*np.exp(-(wave-mu)**2/(2*sigma**2))
            
    def emission_lines(self,wave,params,return_gaussians=False,list_params=False):
        '''
        Constructs the emission lines according to the following rules
        
        -Broad Hbeta gets 4 gaussians
        -Narrow lines get 2 gaussians each
        -Lines used are
            -narrow O II doublet
            -narrow Ne III doublet
            -narrow S II
            -broad H Delta (1 Gaussian)
            -broad H Gamma (1 Gaussian)
            -broad He II (1 Gaussian)
            -broad H Beta
            -narrow H Beta
            -narrow O III 4960
            -narrow O III 5007
        
        - line profile for the 2 narrow Hbeta gaussians are tied to the line profile of the 2 OIII 5007 gaussians
          (meaning WLs, widths, and line ratios are linked. So narrow Hbeta has 1 free parameter)
        - line profile for the 2 narrow OIII 4960 gaussians are tied to the line profile of the 2 OIII 5007 gaussians
          (this has an added restriction that the flux of OIII 4960 should be 1/3 the flux of OIII 5007
           so no free parameters for OIII 4960)
        - Ne III doublet, S II, and O II doublet are all tied to each other in WL, width, and line ratio, but fluxes are free
          So 1 free parameter per line (amplitude), plus 1 line amp ratio, 2 WL and 2 width free parameters for the whole group 
          (because each line gets 2 gaussians)
           Theoretically these should also be tied to OIII 5007, but since we'll be fitting this WL area first,
           and then fitting the Hbeta region after, it makes more sense to tie them seperately
           
        params['ampOII3729']
        params['ampOII3735']
        params['ampNeIII3868']
        params['ampNeIII3966']
        params['ampSII4073']
        params['groupampratio']
        params['groupWLoffset1']
        params['groupWLoffset2']
        params['groupwidth1']
        params['groupwidth2']
        params['ampHDelta']
        params['WLHDelta']
        params['widthHDelta']
        params['ampHGamma']
        params['WLHGamma']
        params['widthHGamma']
        params['ampHeII']
        params['WLHeII']
        params['widthHeII']
        params['ampHBeta1']
        params['WLHBeta1']
        params['widthHBeta1']
        params['ampHBeta2']
        params['WLHBeta2']
        params['widthHBeta2']
        params['ampHBeta3']
        params['WLHBeta3']
        params['widthHBeta3']
        params['ampHBeta4']
        params['WLHBeta4']
        params['widthHBeta4']
        params['ampNHBeta']
        params['ampOIII5007']
        params['group2ampratio']
        params['group2WLoffset1']
        params['group2WLoffset2']
        params['group2width1']
        params['group2width2']

        The gaussians returned when return_gaussians=True are the following (if all lines are chosen in model_select)
        0]  OII 3729 1
        1]  OII 3729 2
        2]  OII 3735 1
        3]  OII 3735 2
        4]  NeIII 3868 1
        5]  NeIII 3868 2
        6]  NeIII 3966 1
        7]  NeIII 3966 2
        8]  SII 4073
        9]  SII 4073
        10] H delta
        11] H gamma
        12] He II
        13] Broad Hbeta 1
        14] Broad Hbeta 2
        15] Broad Hbeta 3
        16] Broad Hbeta 4
        17] Narrow Hbeta 1
        18] Narrow Hbeta 2
        19] OIII 4960 1
        20] OIII 4960 2
        21] OIII 5007 1
        22] OIII 5007 2
        '''
        # when we're using emcee, params need to be provided as a list
        # if that's the case, we'll take the list here and pack it into the appropriate dictionary
        if list_params:
            # self.params_keys is generated inside the fit_emcee function, so params_list and params_keys are indexed the same
            params_dict={}
            for i in range(len(params)):
                params_dict[self.params_keys[i]]=params[i]
                
            # now replace the input params list with the dictionary
            params=params_dict
            
        gaussians={}
        
        # Construct gaussians for narrow lines near 4000 A
        if self.model_select['OII3729']:
            amp_gauss1=params['ampOII3729'] # free parameter for amplitude of this line
            amp_gauss2=params['ampOII3729']*params['groupampratio'] # amp of second gaussian is given by amp ratio
            wave_gauss1=self.OII1_WL+params['groupWLoffset1'] # rest WL + group offset 1
            wave_gauss2=self.OII1_WL+params['groupWLoffset2'] # rest WL + group offset 2
            width_gauss1=params['groupwidth1']
            width_gauss2=params['groupwidth2']
            
            gaussians['OII3729_1']=self.gaussian(wave,amp_gauss1,wave_gauss1,width_gauss1)
            gaussians['OII3729_2']=self.gaussian(wave,amp_gauss2,wave_gauss2,width_gauss2)
        
        if self.model_select['OII3735']:
            amp_gauss1=params['ampOII3735'] # free parameter for amplitude of this line
            amp_gauss2=params['ampOII3735']*params['groupampratio'] # amp of second gaussian is given by amp ratio
            wave_gauss1=self.OII2_WL+params['groupWLoffset1'] # rest WL + group offset 1
            wave_gauss2=self.OII2_WL+params['groupWLoffset2'] # rest WL + group offset 2
            width_gauss1=params['groupwidth1']
            width_gauss2=params['groupwidth2']
            
            gaussians['OII3735_1']=self.gaussian(wave,amp_gauss1,wave_gauss1,width_gauss1)
            gaussians['OII3735_2']=self.gaussian(wave,amp_gauss2,wave_gauss2,width_gauss2)
            
        if self.model_select['NeIII3868']:
            amp_gauss1=params['ampNeIII3868'] # free parameter for amplitude of this line
            amp_gauss2=params['ampNeIII3868']*params['groupampratio'] # amp of second gaussian is given by amp ratio
            wave_gauss1=self.NeIII1_WL+params['groupWLoffset1'] # rest WL + group offset 1
            wave_gauss2=self.NeIII1_WL+params['groupWLoffset2'] # rest WL + group offset 2
            width_gauss1=params['groupwidth1']
            width_gauss2=params['groupwidth2']
            
            gaussians['NeIII3868_1']=self.gaussian(wave,amp_gauss1,wave_gauss1,width_gauss1)
            gaussians['NeIII3868_2']=self.gaussian(wave,amp_gauss2,wave_gauss2,width_gauss2)
            
        if self.model_select['NeIII3966']:
            amp_gauss1=params['ampNeIII3966'] # free parameter for amplitude of this line
            amp_gauss2=params['ampNeIII3966']*params['groupampratio'] # amp of second gaussian is given by amp ratio
            wave_gauss1=self.NeIII2_WL+params['groupWLoffset1'] # rest WL + group offset 1
            wave_gauss2=self.NeIII2_WL+params['groupWLoffset2'] # rest WL + group offset 2
            width_gauss1=params['groupwidth1']
            width_gauss2=params['groupwidth2']
            
            gaussians['NeIII3966_1']=self.gaussian(wave,amp_gauss1,wave_gauss1,width_gauss1)
            gaussians['NeIII3966_2']=self.gaussian(wave,amp_gauss2,wave_gauss2,width_gauss2)
            
        if self.model_select['SII4073']:
            amp_gauss1=params['ampSII4073'] # free parameter for amplitude of this line
            amp_gauss2=params['ampSII4073']*params['groupampratio'] # amp of second gaussian is given by amp ratio
            wave_gauss1=self.SII_WL+params['groupWLoffset1'] # rest WL + group offset 1
            wave_gauss2=self.SII_WL+params['groupWLoffset2'] # rest WL + group offset 2
            width_gauss1=params['groupwidth1']
            width_gauss2=params['groupwidth2']
            
            gaussians['SII4073_1']=self.gaussian(wave,amp_gauss1,wave_gauss1,width_gauss1)
            gaussians['SII4073_2']=self.gaussian(wave,amp_gauss2,wave_gauss2,width_gauss2)
            
        if self.model_select['HDelta']:
            gaussians['HDelta']=self.gaussian(wave,params['ampHDelta'],params['WLHDelta'],params['widthHDelta'])
            
        if self.model_select['HGamma']:
            gaussians['HGamma']=self.gaussian(wave,params['ampHGamma'],params['WLHGamma'],params['widthHGamma'])
            
        if self.model_select['HeII']:
            gaussians['HeII']=self.gaussian(wave,params['ampHeII'],params['WLHeII'],params['widthHeII'])
            
        # Broad H Beta
        gaussians['HBeta_1']=self.gaussian(wave,params['ampHBeta1'],params['WLHBeta1'],params['widthHBeta1'])
        gaussians['HBeta_2']=self.gaussian(wave,params['ampHBeta2'],params['WLHBeta2'],params['widthHBeta2'])
        gaussians['HBeta_3']=self.gaussian(wave,params['ampHBeta3'],params['WLHBeta3'],params['widthHBeta3'])
        gaussians['HBeta_4']=self.gaussian(wave,params['ampHBeta4'],params['WLHBeta4'],params['widthHBeta4'])
        
        # Narrow H Beta
        amp_gauss1=params['ampNHBeta'] # free parameter for amplitude of this line
        amp_gauss2=params['ampNHBeta']*params['group2ampratio'] # amp of second gaussian is given by amp ratio
        wave_gauss1=self.Hbeta_WL+params['group2WLoffset1'] # rest WL + group offset 1
        wave_gauss2=self.Hbeta_WL+params['group2WLoffset2'] # rest WL + group offset 2
        width_gauss1=params['group2width1']
        width_gauss2=params['group2width2']
        
        gaussians['NHBeta_1']=self.gaussian(wave,amp_gauss1,wave_gauss1,width_gauss1)
        gaussians['NHBeta_2']=self.gaussian(wave,amp_gauss2,wave_gauss2,width_gauss2)
        
        # OIII 4960
        amp_gauss1=(1/3)*params['ampOIII5007']
        amp_gauss2=amp_gauss1*params['group2ampratio']
        wave_gauss1=self.OIII1_WL+params['group2WLoffset1']
        wave_gauss2=self.OIII1_WL+params['group2WLoffset2']
        width_gauss1=params['group2width1']
        width_gauss2=params['group2width2']
        
        gaussians['OIII4960_1']=self.gaussian(wave,amp_gauss1,wave_gauss1,width_gauss1)
        gaussians['OIII4960_2']=self.gaussian(wave,amp_gauss2,wave_gauss2,width_gauss2)
        
        # OIII 5007
        amp_gauss1=params['ampOIII5007']
        amp_gauss2=params['ampOIII5007']*params['group2ampratio']
        wave_gauss1=self.OIII2_WL+params['group2WLoffset1']
        wave_gauss2=self.OIII2_WL+params['group2WLoffset2']
        width_gauss1=params['group2width1']
        width_gauss2=params['group2width2']
        
        gaussians['OIII5007_1']=self.gaussian(wave,amp_gauss1,wave_gauss1,width_gauss1)
        gaussians['OIII5007_2']=self.gaussian(wave,amp_gauss2,wave_gauss2,width_gauss2)

        if return_gaussians:
            return gaussians
            
        emission=np.zeros(len(wave))
        for key in gaussians:
            emission+=gaussians[key]
            
        return emission
            
    def power_law(self,wave,norm,slope):
        '''
        continuum power law. Normalized at a specified WL.
        
        params:
            wave - wavelengths of data
            norm - power-law normalization factor
            slope - power-law slope
        '''
        
        return norm*(wave/self.pl_norm_wl)**slope
    
    def construct_model(self,wave,params,fit_type):
        '''
        constructs the spectral decomposition model
        
        params['pl_norm']        - power law normalization
        params['pl_alpha']       - power law slope
        params['fe_ii_width']    - Fe II convolution width index
        params['fe_ii_scale']    - Fe II scale factor
        params['host_age']       - Host galaxy stellar age index
        params['host_scale']     - Host galaxy scale factor
        params['bal_cont_tau']   - Balmer continuum tau index
        params['bal_cont_scale'  - Balmer continuum scale factor
        params['bal_high_width'] - Balmer high order FWHM index
        params['bal_high_scale'] - Balmer high order scale factor
        params[10-] - emission line parameters (check emission_lines() function)
        '''
        model=np.zeros(len(wave))
        if self.model_select['PowerLaw']:
            power_law=self.power_law(wave,params['pl_norm'],params['pl_alpha'])
            model=model+power_law
        if self.model_select['FeII']:
            fe_ii_op=self.template_fitter(wave,self.feii_templates,params['fe_ii_width'],params['fe_ii_scale'])
            model=model+fe_ii_op
        if self.model_select['Host']:
            host=self.template_fitter(wave,self.stellar_templates,params['host_age'],params['host_scale'])
            model=model+host
        if self.model_select['BalmerCont']:
            balmer_cont=self.template_fitter(wave,self.balmer_cont_templates,params['bal_cont_tau'],params['bal_cont_scale'])
            model=model+balmer_cont
        if self.model_select['BalmerHigh']:
            balmer_highorder=self.template_fitter(wave,self.balmer_highorder_templates,params['bal_high_width'],params['bal_high_scale'])
            model=model+balmer_highorder
            
        if fit_type=='cont':
            return model
        elif fit_type=='full': 
            emission=self.emission_lines(wave,params)
            model=model+emission
            return model
        else:
            raise ValueError("Fit Type Must Be 'full' or 'cont', not '{:s}'".format(fit_type))
    
    def guess_initial_params(self):
        '''
        Uses the spectrum to guess initial values for model params
        '''
        
        # set power law normalization equal to flux at pl_norm_wl
        
        # first handle case where pl_norm_wl not in bandpass
        if np.amin(self.wave)>self.pl_norm_wl:
            # take first index
            idx_low=0
        else:
            for i in range(len(self.wave)):
                if self.wave[i]>=self.pl_norm_wl:
                    idx_low=i
                    break
        pl_norm=self.flux[idx_low]
        
        # check flux at further wavelength, use formula for power law to set power law slope
        
        # first handle case where 5500 not in bandpass
        if np.amax(self.wave)<5500:
            # take last index
            idx_5500=-1
        else:
            for i in range(idx_low,len(self.wave)):
                if self.wave[i]>=5500:
                    idx_5500=i
                    break
                
        flux_5500=self.flux[idx_5500]
        alpha=(np.log(flux_5500)-np.log(pl_norm))/(np.log(5500/self.pl_norm_wl))
        if np.isnan(alpha):
            alpha=-1.8
        
        params_init={}
        
        if self.model_select['PowerLaw']:
            params_init['pl_norm']=pl_norm-2
            params_init['pl_alpha']=alpha
        if self.model_select['FeII']:
            params_init['fe_ii_width']=30
            params_init['fe_ii_scale']=1
        if self.model_select['Host']:
            params_init['host_age']=4
            params_init['host_scale']=1
        if self.model_select['BalmerCont']:
            params_init['bal_cont_tau']=10
            params_init['bal_cont_scale']=1
        if self.model_select['BalmerHigh']:
            params_init['bal_high_width']=10
            params_init['bal_high_scale']=1
        
        # guess amplitudes of emission lines based on flux at center wavelength - power law
        pl_init=self.power_law(self.wave,pl_norm,alpha)
        
        # Initialize narrow lines around 4000 A with WL offsets of 0, narrow width, and amplitude of 1
        narrow_4000=False # are we using any narrow lines near 4000 A?
        if self.model_select['OII3729']:
            params_init['ampOII3729']=1
            narrow_4000=True
        if self.model_select['OII3735']:
            params_init['ampOII3735']=1
            narrow_4000=True
        if self.model_select['NeIII3868']:
            params_init['ampNeIII3868']=1
            narrow_4000=True
        if self.model_select['NeIII3966']:
            params_init['ampNeIII3966']=1
            narrow_4000=True
        if self.model_select['SII4073']:
            params_init['ampSII4073']=1
            narrow_4000=True
        if narrow_4000:
            params_init['groupampratio']=1
            params_init['groupWLoffset1']=0
            params_init['groupWLoffset2']=0
            params_init['groupwidth1']=self.narrow_start
            params_init['groupwidth2']=self.narrow_start
        
        # Guess amplitude of H Delta, H Gamma, H Beta, and OIII 5007
        # based on flux at center wavelength - power law at that wavelength
        
        # H Delta
        if self.model_select['HDelta']:
            idx_HDelta=np.where(self.wave>self.Hdelta_WL)[0][0]
            amp_HDelta=self.flux[idx_HDelta]-pl_init[idx_HDelta]
            params_init['ampHDelta']=amp_HDelta
            params_init['WLHDelta']=self.Hdelta_WL
            params_init['widthHDelta']=self.broad_start
        
        # H Gamma
        if self.model_select['HGamma']:
            idx_HGamma=np.where(self.wave>self.Hgamma_WL)[0][0]
            amp_HGamma=self.flux[idx_HGamma]-pl_init[idx_HGamma]
            params_init['ampHGamma']=amp_HGamma
            params_init['WLHGamma']=self.Hgamma_WL
            params_init['widthHGamma']=self.broad_start
        
        # He II
        if self.model_select['HeII']:
            idx_HeII=np.where(self.wave>self.HeII_WL)[0][0]
            amp_HeII=self.flux[idx_HeII]-pl_init[idx_HeII]
            params_init['ampHeII']=amp_HeII
            params_init['WLHeII']=self.HeII_WL
            params_init['widthHeII']=self.broad_start
        
        # Broad H Beta
        idx_HBeta=np.where(self.wave>self.Hbeta_WL)[0][0]
        amp_HBeta=self.flux[idx_HBeta]-pl_init[idx_HBeta]
        params_init['ampHBeta1']=amp_HBeta/4
        params_init['WLHBeta1']=self.Hbeta_WL-15
        params_init['widthHBeta1']=self.broad_start
        
        params_init['ampHBeta2']=amp_HBeta/4
        params_init['WLHBeta2']=self.Hbeta_WL-5
        params_init['widthHBeta2']=self.broad_start
        
        params_init['ampHBeta3']=amp_HBeta/4
        params_init['WLHBeta3']=self.Hbeta_WL+5
        params_init['widthHBeta3']=self.broad_start
        
        params_init['ampHBeta4']=amp_HBeta/4
        params_init['WLHBeta4']=self.Hbeta_WL+15
        params_init['widthHBeta4']=self.broad_start
        
        # Narrow H Beta
        params_init['ampNHBeta']=1
        
        #OIII 5007
        idx_OIII2=np.where(self.wave>self.OIII2_WL)[0][0]
        amp_OIII2=self.flux[idx_OIII2]-pl_init[idx_OIII2]
        params_init['ampOIII5007']=amp_OIII2/2
        params_init['group2ampratio']=1
        params_init['group2WLoffset1']=0
        params_init['group2WLoffset2']=0
        params_init['group2width1']=self.narrow_start
        params_init['group2width2']=self.narrow_start

        self.params_init=params_init
        
    def residuals(self,params,data,fit_type='full'):
        '''
        residuals function for kmpfit. Returns (flux-model)/err, where model is constructed from params.

        params - dictionary of parameter values
        data - tuple with (wave,flux,err)
        
        '''
        
        wave,flux,err=data

        model=self.construct_model(wave,params,fit_type)
        
        resid=(flux-model)/err
        #resid_nonan=np.where(resid==np.nan,10,resid)
        
        return resid
    
    
    def fit(self,minimize_method='powell',verbose=False):
        '''
        Fits the spectrum given initial parameters (possibly generated by guess_initial_params())
        This fitter performs the following steps
        
            - First fits Power Law and Fe II only (other continuum params fixed), over select user specified WL regions
            - Using Power Law and Fe II solution from first fit (and holding these params fixed), 
              then fits host and narrow lines near 4000 A, over select user specified WL regions
            - Fits H Delta and H Gamma emission lines (all other params fixed)
            - Fits He II, Broad H Beta, Narrow H Beta, OIII 4960, and OIII 5007 (all other params fixed)
              (remember OIII 4960 has no free parameters, Narrow H Beta has only one)
            - Finally fits all parameters 
        '''
        time0=time.time()
        
        
        # Fit 1: Power Law + Fe II only (fitting over user specified WL regions)
        indexes=self.wave_region_select(self.wave,self.pl_feii_wls)
        data_fit1=(self.wave[indexes],self.flux[indexes],self.err[indexes])
        
        fit_params=Parameters()
        # Power Law and Fe II can vary
        if self.model_select['PowerLaw']:
            fit_params.add('pl_norm',
                           value=self.params_init['pl_norm'],
                           min=self.lower_bounds['pl_norm'],
                           max=self.upper_bounds['pl_norm'],
                           vary=True)
            fit_params.add('pl_alpha',
                           value=self.params_init['pl_alpha'],
                           min=self.lower_bounds['pl_alpha'],
                           max=self.upper_bounds['pl_alpha'],
                           vary=True)
        if self.model_select['FeII']:
            fit_params.add('fe_ii_width',
                           value=self.params_init['fe_ii_width'],
                           min=self.lower_bounds['fe_ii_width'],
                           max=self.upper_bounds['fe_ii_width'],
                           vary=True)
            fit_params.add('fe_ii_scale',
                           value=self.params_init['fe_ii_scale'],
                           min=self.lower_bounds['fe_ii_scale'],
                           max=self.upper_bounds['fe_ii_scale'],
                           vary=True)
        # Fix other continuum parameters
        if self.model_select['Host']:
            fit_params.add('host_age',
                           value=self.params_init['host_age'],
                           min=self.lower_bounds['host_age'],
                           max=self.upper_bounds['host_age'],
                           vary=False)
            fit_params.add('host_scale',
                           value=self.params_init['host_scale'],
                           min=self.lower_bounds['host_scale'],
                           max=self.upper_bounds['host_scale'],
                           vary=False)
        if self.model_select['BalmerCont']:
            fit_params.add('bal_cont_tau',
                           value=self.params_init['bal_cont_tau'],
                           min=self.lower_bounds['bal_cont_tau'],
                           max=self.upper_bounds['bal_cont_tau'],
                           vary=False)
            fit_params.add('bal_cont_scale',
                           value=self.params_init['bal_cont_scale'],
                           min=self.lower_bounds['bal_cont_scale'],
                           max=self.upper_bounds['bal_cont_scale'],
                           vary=False)
        if self.model_select['BalmerHigh']:
            fit_params.add('bal_high_width',
                           value=self.params_init['bal_high_width'],
                           min=self.lower_bounds['bal_high_width'],
                           max=self.upper_bounds['bal_high_width'],
                           vary=False)
            fit_params.add('bal_high_scale',
                           value=self.params_init['bal_high_scale'],
                           min=self.lower_bounds['bal_high_scale'],
                           max=self.upper_bounds['bal_high_scale'],
                           vary=False)
        
        fit1_result=minimize(self.residuals, 
                             fit_params, 
                             args=(data_fit1,),
                             kws={'fit_type': 'cont'},
                             nan_policy=self.nan_policy,
                             method=minimize_method,
                             max_nfev=None)
        
        fit_params=fit1_result.params
        self.params_fit1=fit1_result.params
        
        # Fit 2: Host and Narrow Lines around 4000 A, also Balmer (all other parameters fixed)
        indexes=self.wave_region_select(self.wave,self.host_narrow_wls)
        data_fit2=(self.wave[indexes],self.flux[indexes],self.err[indexes])
        
        # First update status of continuum parameters
        if self.model_select['PowerLaw']:
            fit_params['pl_norm'].set(vary=False)
            fit_params['pl_alpha'].set(vary=False)
        if self.model_select['FeII']:
            fit_params['fe_ii_width'].set(vary=False)
            fit_params['fe_ii_scale'].set(vary=False)
        if self.model_select['Host']:
            fit_params['host_age'].set(vary=True)
            fit_params['host_scale'].set(vary=True)
        if self.model_select['BalmerCont']:
            fit_params['bal_cont_tau'].set(vary=True)
            fit_params['bal_cont_scale'].set(vary=True)
        if self.model_select['BalmerHigh']:
            fit_params['bal_high_width'].set(vary=True)
            fit_params['bal_high_scale'].set(vary=True)
        
        
        # Now add parameters for all emission lines
        # Narrow lines around 4000 A are allowed to vary
        fit_4000_narrow_lines=False
        if self.model_select['OII3729']:
            fit_params.add('ampOII3729',
                           value=self.params_init['ampOII3729'],
                           min=self.lower_bounds['ampOII3729'],
                           max=self.upper_bounds['ampOII3729'],
                           vary=True)
            fit_4000_narrow_lines=True
        if self.model_select['OII3735']:
            fit_params.add('ampOII3735',
                           value=self.params_init['ampOII3735'],
                           min=self.lower_bounds['ampOII3735'],
                           max=self.upper_bounds['ampOII3735'],
                           vary=True)
            fit_4000_narrow_lines=True
        if self.model_select['NeIII3868']:
            fit_params.add('ampNeIII3868',
                           value=self.params_init['ampNeIII3868'],
                           min=self.lower_bounds['ampNeIII3868'],
                           max=self.upper_bounds['ampNeIII3868'],
                           vary=True)
            fit_4000_narrow_lines=True
        if self.model_select['NeIII3966']:
            fit_params.add('ampNeIII3966',
                           value=self.params_init['ampNeIII3966'],
                           min=self.lower_bounds['ampNeIII3966'],
                           max=self.upper_bounds['ampNeIII3966'],
                           vary=True)
            fit_4000_narrow_lines=True
        if self.model_select['SII4073']:
            fit_params.add('ampSII4073',
                           value=self.params_init['ampSII4073'],
                           min=self.lower_bounds['ampSII4073'],
                           max=self.upper_bounds['ampSII4073'],
                           vary=True)
            fit_4000_narrow_lines=True
        if fit_4000_narrow_lines:
            fit_params.add('groupampratio',
                           value=self.params_init['groupampratio'],
                           min=self.lower_bounds['groupampratio'],
                           max=self.upper_bounds['groupampratio'],
                           vary=True)
            fit_params.add('groupWLoffset1',
                           value=self.params_init['groupWLoffset1'],
                           min=self.lower_bounds['groupWLoffset1'],
                           max=self.upper_bounds['groupWLoffset1'],
                           vary=True)
            fit_params.add('groupWLoffset2',
                           value=self.params_init['groupWLoffset2'],
                           min=self.lower_bounds['groupWLoffset2'],
                           max=self.upper_bounds['groupWLoffset2'],
                           vary=True)
            fit_params.add('groupwidth1',
                           value=self.params_init['groupwidth1'],
                           min=self.lower_bounds['groupwidth1'],
                           max=self.upper_bounds['groupwidth1'],
                           vary=True)
            fit_params.add('groupwidth2',
                           value=self.params_init['groupwidth2'],
                           min=self.lower_bounds['groupwidth2'],
                           max=self.upper_bounds['groupwidth2'],
                           vary=True)
        
        # Other lines are fixed
        if self.model_select['HDelta']:
            fit_params.add('ampHDelta',
                           value=self.params_init['ampHDelta'],
                           min=self.lower_bounds['ampHDelta'],
                           max=self.upper_bounds['ampHDelta'],
                           vary=False)
            fit_params.add('WLHDelta',
                           value=self.params_init['WLHDelta'],
                           min=self.lower_bounds['WLHDelta'],
                           max=self.upper_bounds['WLHDelta'],
                           vary=False)
            fit_params.add('widthHDelta',
                           value=self.params_init['widthHDelta'],
                           min=self.lower_bounds['widthHDelta'],
                           max=self.upper_bounds['widthHDelta'],
                           vary=False)
        if self.model_select['HGamma']:
            fit_params.add('ampHGamma',
                           value=self.params_init['ampHGamma'],
                           min=self.lower_bounds['ampHGamma'],
                           max=self.upper_bounds['ampHGamma'],
                           vary=False)
            fit_params.add('WLHGamma',
                           value=self.params_init['WLHGamma'],
                           min=self.lower_bounds['WLHGamma'],
                           max=self.upper_bounds['WLHGamma'],
                           vary=False)
            fit_params.add('widthHGamma',
                           value=self.params_init['widthHGamma'],
                           min=self.lower_bounds['widthHGamma'],
                           max=self.upper_bounds['widthHGamma'],
                           vary=False)
        if self.model_select['HeII']:
            fit_params.add('ampHeII',
                           value=self.params_init['ampHeII'],
                           min=self.lower_bounds['ampHeII'],
                           max=self.upper_bounds['ampHeII'],
                           vary=False)
            fit_params.add('WLHeII',
                           value=self.params_init['WLHeII'],
                           min=self.lower_bounds['WLHeII'],
                           max=self.upper_bounds['WLHeII'],
                           vary=False)
            fit_params.add('widthHeII',
                           value=self.params_init['widthHeII'],
                           min=self.lower_bounds['widthHeII'],
                           max=self.upper_bounds['widthHeII'],
                           vary=False)
        
        fit_params.add('ampHBeta1',
                       value=self.params_init['ampHBeta1'],
                       min=self.lower_bounds['ampHBeta1'],
                       max=self.upper_bounds['ampHBeta1'],
                       vary=False)
        fit_params.add('WLHBeta1',
                       value=self.params_init['WLHBeta1'],
                       min=self.lower_bounds['WLHBeta1'],
                       max=self.upper_bounds['WLHBeta1'],
                       vary=False)
        fit_params.add('widthHBeta1',
                       value=self.params_init['widthHBeta1'],
                       min=self.lower_bounds['widthHBeta1'],
                       max=self.upper_bounds['widthHBeta1'],
                       vary=False)
        fit_params.add('ampHBeta2',
                       value=self.params_init['ampHBeta2'],
                       min=self.lower_bounds['ampHBeta2'],
                       max=self.upper_bounds['ampHBeta2'],
                       vary=False)
        fit_params.add('WLHBeta2',
                       value=self.params_init['WLHBeta2'],
                       min=self.lower_bounds['WLHBeta2'],
                       max=self.upper_bounds['WLHBeta2'],
                       vary=False)
        fit_params.add('widthHBeta2',
                       value=self.params_init['widthHBeta2'],
                       min=self.lower_bounds['widthHBeta2'],
                       max=self.upper_bounds['widthHBeta2'],
                       vary=False)
        fit_params.add('ampHBeta3',
                       value=self.params_init['ampHBeta3'],
                       min=self.lower_bounds['ampHBeta3'],
                       max=self.upper_bounds['ampHBeta3'],
                       vary=False)
        fit_params.add('WLHBeta3',
                       value=self.params_init['WLHBeta3'],
                       min=self.lower_bounds['WLHBeta3'],
                       max=self.upper_bounds['WLHBeta3'],
                       vary=False)
        fit_params.add('widthHBeta3',
                       value=self.params_init['widthHBeta3'],
                       min=self.lower_bounds['widthHBeta3'],
                       max=self.upper_bounds['widthHBeta3'],
                       vary=False)
        fit_params.add('ampHBeta4',
                       value=self.params_init['ampHBeta4'],
                       min=self.lower_bounds['ampHBeta4'],
                       max=self.upper_bounds['ampHBeta4'],
                       vary=False)
        fit_params.add('WLHBeta4',
                       value=self.params_init['WLHBeta4'],
                       min=self.lower_bounds['WLHBeta4'],
                       max=self.upper_bounds['WLHBeta4'],
                       vary=False)
        fit_params.add('widthHBeta4',
                       value=self.params_init['widthHBeta4'],
                       min=self.lower_bounds['widthHBeta4'],
                       max=self.upper_bounds['widthHBeta4'],
                       vary=False)
        
        fit_params.add('ampNHBeta',
                       value=self.params_init['ampNHBeta'],
                       min=self.lower_bounds['ampNHBeta'],
                       max=self.upper_bounds['ampNHBeta'],
                       vary=False)
        fit_params.add('ampOIII5007',
                       value=self.params_init['ampOIII5007'],
                       min=self.lower_bounds['ampOIII5007'],
                       max=self.upper_bounds['ampOIII5007'],
                       vary=False)
        fit_params.add('group2ampratio',
                       value=self.params_init['group2ampratio'],
                       min=self.lower_bounds['group2ampratio'],
                       max=self.upper_bounds['group2ampratio'],
                       vary=False)
        fit_params.add('group2WLoffset1',
                       value=self.params_init['group2WLoffset1'],
                       min=self.lower_bounds['group2WLoffset1'],
                       max=self.upper_bounds['group2WLoffset1'],
                       vary=False)
        fit_params.add('group2WLoffset2',
                       value=self.params_init['group2WLoffset2'],
                       min=self.lower_bounds['group2WLoffset2'],
                       max=self.upper_bounds['group2WLoffset2'],
                       vary=False)
        fit_params.add('group2width1',
                       value=self.params_init['group2width1'],
                       min=self.lower_bounds['group2width1'],
                       max=self.upper_bounds['group2width1'],
                       vary=False)
        fit_params.add('group2width2',
                       value=self.params_init['group2width2'],
                       min=self.lower_bounds['group2width2'],
                       max=self.upper_bounds['group2width2'],
                       vary=False)
        
        fit2_result=minimize(self.residuals, 
                             fit_params, 
                             args=(data_fit2,),
                             kws={'fit_type': 'full'},
                             nan_policy=self.nan_policy,
                             method=minimize_method,
                             max_nfev=None)
        fit_params=fit2_result.params
        self.params_fit2=fit2_result.params
        
        # Fit 3: H Delta and H Gamma, all other params fixed
        indexes=self.wave_region_select(self.wave,self.hdelta_hgamma_wls)
        data_fit3=(self.wave[indexes],self.flux[indexes],self.err[indexes])
        
        # First set all params to not vary
        param_names=fit_params.valuesdict().keys()
        for i in param_names: 
            fit_params[i].set(vary=False)
        
        # Then set proper params to vary
        if self.model_select['HDelta']:
            fit_params['ampHDelta'].set(vary=True)
            fit_params['WLHDelta'].set(vary=True)
            fit_params['widthHDelta'].set(vary=True)
        if self.model_select['HGamma']:
            fit_params['ampHGamma'].set(vary=True)
            fit_params['WLHGamma'].set(vary=True)
            fit_params['widthHGamma'].set(vary=True)
        
        fit3_result=minimize(self.residuals, 
                             fit_params, 
                             args=(data_fit3,),
                             kws={'fit_type': 'full'},
                             nan_policy=self.nan_policy,
                             method=minimize_method,
                             max_nfev=None)
        
        fit_params=fit3_result.params
        self.params_fit3=fit3_result.params
        
        # Fit 4: Broad H Beta, Narrow H Beta, OIII 4960, OIII 5007, all other params fixed
        indexes=self.wave_region_select(self.wave,self.hbeta_oiii_wls)
        data_fit4=(self.wave[indexes],self.flux[indexes],self.err[indexes])
        
        # First set all params to not vary
        param_names=fit_params.valuesdict().keys()
        for i in param_names: 
            fit_params[i].set(vary=False)
            
        # Then set proper params to vary
        if self.model_select['HeII']:
            fit_params['ampHeII'].set(vary=True)
            fit_params['WLHeII'].set(vary=True)
            fit_params['widthHeII'].set(vary=True)
        fit_params['ampHBeta1'].set(vary=True)
        fit_params['WLHBeta1'].set(vary=True)
        fit_params['widthHBeta1'].set(vary=True)
        fit_params['ampHBeta2'].set(vary=True)
        fit_params['WLHBeta2'].set(vary=True)
        fit_params['widthHBeta2'].set(vary=True)
        fit_params['ampHBeta3'].set(vary=True)
        fit_params['WLHBeta3'].set(vary=True)
        fit_params['widthHBeta3'].set(vary=True)
        fit_params['ampHBeta4'].set(vary=True)
        fit_params['WLHBeta4'].set(vary=True)
        fit_params['widthHBeta4'].set(vary=True)
        fit_params['ampNHBeta'].set(vary=True)
        fit_params['ampOIII5007'].set(vary=True)
        fit_params['group2ampratio'].set(vary=True)
        fit_params['group2WLoffset1'].set(vary=True)
        fit_params['group2WLoffset2'].set(vary=True)
        fit_params['group2width1'].set(vary=True)
        fit_params['group2width2'].set(vary=True)
        
        fit4_result=minimize(self.residuals, 
                             fit_params, 
                             args=(data_fit4,),
                             kws={'fit_type': 'full'},
                             nan_policy=self.nan_policy,
                             method=minimize_method,
                             max_nfev=None)
        
        fit_params=fit4_result.params
        self.params_fit4=fit4_result.params
        
        # Fit 5: All parameters can vary
        data_fit5=(self.wave,self.flux,self.err)
        
        param_names=fit_params.valuesdict().keys()
        for i in param_names: 
            fit_params[i].set(vary=True)
            
        result=minimize(self.residuals, 
                        fit_params, 
                        args=(data_fit5,),
                        kws={'fit_type': 'full'},
                        nan_policy=self.nan_policy,
                        method=minimize_method,
                        max_nfev=None)
        
        fit_params=result.params
        
        self.params_final=result.params
        self.results_final=result
        self.chisq=result.chisqr
        self.red_chisq=result.redchi
        self.aic=result.aic
        self.bic=result.bic
        if self.model_select['PowerLaw']:
            self.p_law=self.power_law(self.wave,self.params_final['pl_norm'],self.params_final['pl_alpha'])
        if self.model_select['FeII']:
            self.fe_ii=self.template_fitter(self.wave,self.feii_templates,self.params_final['fe_ii_width'],self.params_final['fe_ii_scale'])
        if self.model_select['Host']:
            self.host=self.template_fitter(self.wave,self.stellar_templates,self.params_final['host_age'],self.params_final['host_scale'])
        if self.model_select['BalmerCont']:
            self.bal_cont=self.template_fitter(self.wave,
                                               self.balmer_cont_templates,
                                               self.params_final['bal_cont_tau'],
                                               self.params_final['bal_cont_scale'])
        if self.model_select['BalmerHigh']:
            self.bal_high=self.template_fitter(self.wave,
                                               self.balmer_highorder_templates,
                                               self.params_final['bal_high_width'],
                                               self.params_final['bal_high_scale'])
        
        self.emission=self.emission_lines(self.wave,self.params_final)
        
        time1=time.time()
        
        if verbose:
            print('Fitting Time: {:.1f} s'.format(time1-time0))
        
    def construct_model_emcee(self,params_list):
        '''
        emcee passes a list of params (not dict), so our prior construct model function won't work.
        Writing a similar function here that will just work with a list
        
                
        params[0]   - power law normalization
        params[1]   - power law slope
        params[2]   - Fe II convolution width index
        params[3]   - Fe II scale factor
        params[4]   - Host galaxy stellar age index
        params[5]   - Host galaxy scale factor
        params[6]   - Balmer continuum tau index
        params[7]   - Balmer continuum scale factor
        params[8]   - Balmer high order FWHM index
        params[9]   - Balmer high order scale factor
        params[10-] - emission line parameters (we'll feed the emission_lines function the whole list bc it makes certain things easier)
        '''
        
        model=np.zeros(len(self.wave))
        if self.model_select['PowerLaw']:
            power_law=self.power_law(self.wave,params_list[self.pl_norm_idx],params_list[self.pl_alpha_idx])
            model=model+power_law
        if self.model_select['FeII']:
            fe_ii_op=self.template_fitter(self.wave,self.feii_templates,params_list[self.fe_width_idx],params_list[self.fe_scale_idx])
            model=model+fe_ii_op
        if self.model_select['Host']:
            host=self.template_fitter(self.wave,self.stellar_templates,params_list[self.host_age_idx],params_list[self.host_scale_idx])
            model=model+host
        if self.model_select['BalmerCont']:
            balmer_cont=self.template_fitter(self.wave,self.balmer_cont_templates,params_list[self.cont_tau_idx],params_list[self.cont_scale_idx])
            model=model+balmer_cont
        if self.model_select['BalmerHigh']:
            balmer_highorder=self.template_fitter(self.wave,self.balmer_highorder_templates,params_list[self.high_width_idx],params_list[self.high_scale_idx])
            model=model+balmer_highorder
        
        emission=self.emission_lines(self.wave,params_list,return_gaussians=False,list_params=True)
        model=model+emission
        
        return model
   
    def log_likelihood(self,params_list):
        '''
        log likelihood estimation for emcee
        '''
        
        model=self.construct_model_emcee(params_list)

        
        return -0.5 * np.sum((self.flux - model) ** 2 / self.err**2)
    
    def log_prior(self,params_list):
        '''
        uniform priors for emcee
        '''
        
        for i in range(len(params_list)):
            
            key=self.params_keys[i]
            
            if params_list[i]<self.lower_bounds[key]:
                return -np.inf
            if params_list[i]>self.upper_bounds[key]:
                return -np.inf
        return 0.0
    
    
    def fit_emcee(self,nwalkers=100,nsteps=5000):
        '''
        Fits the spectrum given initial parameters (possibly generated by guess_initial_params())
        This function uses MCMC to model the whole spectrum, it does not model portions of the spectrum with some params fixed
        IMPORTANT: Currently this uses lower_bounds and upper_bounds to define uniform priors,
            it does not have a way of specifying other priors currently.
            
        This will return a few additional things, like
            - posterior chain
            - acceptance fraction per walker
            - chain for each model param
            
        You can use the model param chain to estimate uncertainties
        
        params:
            -params_init - initial parameter values
            -lower_bounds - lower limit for each parameter
            -upper_bounds - upper limit for each parameter
            -nwalkers - number of emcee walkers (int)
            -nsteps - number of iterations for MCMC (int)
        '''
        
        def log_probability(params_list):
            '''
            posterior for emcee
            '''

            lp=self.log_prior(params_list)
            if not np.isfinite(lp):
                return -np.inf
            return lp + self.log_likelihood(params_list)
            
        
        params_list=np.asarray([])
        params_keys=[]
        for key,value in self.params_init.items():
            params_keys.append(key)
            params_list=np.append(params_list,value)
            
        self.params_keys=params_keys
            
        # find indexes for continuum params in params_list
        if self.model_select['PowerLaw']:
            self.pl_norm_idx=self.params_keys.index('pl_norm')
            self.pl_alpha_idx=self.params_keys.index('pl_alpha')
        if self.model_select['FeII']:
            self.fe_width_idx=self.params_keys.index('fe_ii_width')
            self.fe_scale_idx=self.params_keys.index('fe_ii_scale')
        if self.model_select['Host']:
            self.host_age_idx=self.params_keys.index('host_age')
            self.host_scale_idx=self.params_keys.index('host_scale')
        if self.model_select['BalmerCont']:
            self.cont_tau_idx=self.params_keys.index('bal_cont_tau')
            self.cont_scale_idx=self.params_keys.index('bal_cont_scale')
        if self.model_select['BalmerHigh']:
            self.high_width_idx=self.params_keys.index('bal_high_width')
            self.high_scale_idx=self.params_keys.index('bal_high_scale')
            
        self.params_keys=params_keys
        
        ndim=len(params_list)
        
        # Initialize walkers in a small Gaussian ball around initial parameter values
        pos=params_list+ 1e-4 * np.random.randn(nwalkers, ndim)
        
        # Run Emcee
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
        sampler.run_mcmc(pos,nsteps,progress=True)
        
        self.sampler=sampler
    
    
if __name__=='__main__':
    
    data=np.load('./files/spec-9141-57663-0789_unred_z.npy',allow_pickle=True)
    WL=data.item().get('WL')
    INT=data.item().get('INT')
    ERR=data.item().get('ERR')
    
    spec=SpecDecomp(WL, INT, ERR)
    
    minWL=3750       # minimum wavelength for fitting
    maxWL=5550       # maximum wavelength for fitting
    spec.clip_data_wavelength(minWL,maxWL)
    
    # Dictionary with booleans for whether to include certain model components
    model_select={}
    model_select['PowerLaw']=True
    model_select['BalmerCont']=True
    model_select['BalmerHigh']=True
    model_select['FeII']=True
    model_select['Host']=True
    model_select['OII3729']=True
    model_select['OII3735']=True
    model_select['NeIII3868']=True
    model_select['NeIII3966']=True
    model_select['SII4073']=True
    model_select['HDelta']=True
    model_select['HGamma']=True
    model_select['HeII']=True
    
    spec.set_model(model_select)
    
    # gradient descent-ish fitting example
    print('Starting Gradient Descent Fit')
    spec.guess_initial_params()
    #spec.fit()
    
    # MCMC (emcee) fitting example
    print('Starting emcee Fit')
    spec.fit_emcee(nwalkers=100,nsteps=100)
    sampler=spec.sampler