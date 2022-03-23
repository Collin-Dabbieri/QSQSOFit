import math
import numpy as np
from matplotlib import pyplot as plt
import os
import glob
from lmfit import Parameters, fit_report, minimize
import time
import emcee

OII1_WL=3729.225
OII2_WL=3735.784
NeIII1_WL=3868.265
NeIII2_WL=3966.244
SII_WL=4073.125
Hdelta_WL=4102
Hgamma_WL=4341

HeII_WL=4685

Hbeta_WL=4862.721
OIII1_WL=4960.295
OIII2_WL=5008.239

####################################################################################################################################
############# USER SPECIFIED VALUES
####################################################################################################################################

feii_template_path='./files/fe_op_templates.npy'
stellar_template_path='./files/PyQSOfit_MILES_templates.dat'
balmer_cont_template_path='./files/balmer_cont_templates.npy'
balmer_highorder_template_path='./files/balmer_highorder_templates.npy'

num_templates_feii=100
num_templates_host=8
num_templates_balmer_cont=20
num_templates_balmer_highorder=21

min_WL=3750       # minimum wavelength for fitting
max_WL=5550       # maximum wavelength for fitting


pl_feii_wls=[[4150,4250],[4450,4800],[5050,5750]] # list of lists, gives WL regions for initial fit w only power law and Fe II
host_narrow_wls=[[3750,4080]] # list of lists, gives WL regions for host + narrow line fits near 4000 A
hdelta_hgamma_wls=[[4000,4500]] # list of lists, gives WL regions for H Delta and H Gamma fits
hbeta_oiii_wls=[[4600,5100]]  # list of lists, gives WL regions for He II, H Beta and O III fits


pl_norm_wl=4000   # Wavelength at which the power law is normalized
        
broad_start=15    # starting width for broad lines
narrow_start=5    # starting width for narrow lines

broad_max=200     # max width for broad lines
narrow_max=50     # max width for narrow lines

max_offset=70     # max offset for center of emission line
max_amp=1000       # max amplitude of gaussian emission lines

minimize_method=['least_squares','powell'] # see https://lmfit.github.io/lmfit-py/fitting.html for list of fitting methods
# TODO if minimize_method='emcee' then the fitting algorithm will perform only one fit with all of the parameters
# TODO if minimize_method is a list of methods, then perform fit with each method, choose method with the best chi squared

nan_policy='raise'


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
# params['ampOII3729']
# params['ampOII3735']
# params['ampNeIII3868']
# params['ampNeIII3966']
# params['ampSII4073']
# params['groupampratio']
# params['groupWLoffset1']
# params['groupWLoffset2']
# params['groupwidth1']
# params['groupwidth2']
# params['ampHDelta']
# params['WLHDelta']
# params['widthHDelta']
# params['ampHGamma']
# params['WLHGamma']
# params['widthHGamma']

# params['ampHeII']
# params['WLHeII']
# params['widthHeII']

# params['ampHBeta1']
# params['WLHBeta1']
# params['widthHBeta1']
# params['ampHBeta2']
# params['WLHBeta2']
# params['widthHBeta2']
# params['ampHBeta3']
# params['WLHBeta3']
# params['widthHBeta3']
# params['ampHBeta4']
# params['WLHBeta4']
# params['widthHBeta4']
# params['ampNHBeta']
# params['ampOIII50071']
# params['WLOIII50071']
# params['widthOIII50071']
# params['ampOIII50072']
# params['WLOIII50072']
# params['widthOIII50072']

lower_bounds=[0,-10,0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,-5,-5,0,0,
              0,Hdelta_WL-max_offset,0,0,Hgamma_WL-max_offset,0,
              0,HeII_WL-max_offset,0,
              0,Hbeta_WL-max_offset,0,0,Hbeta_WL-max_offset,0,
              0,Hbeta_WL-max_offset,0,0,Hbeta_WL-max_offset,0,
              0,
              0,OIII2_WL-max_offset,0,0,OIII2_WL-max_offset,0]

upper_bounds=[500,4,
              num_templates_feii-1.1,200,
              num_templates_host-1.1,200,
              num_templates_balmer_cont-1.1,200,
              num_templates_balmer_highorder-1.1,200,
              max_amp,max_amp,max_amp,max_amp,max_amp,100,5,5,narrow_max,narrow_max,
              max_amp,Hdelta_WL+max_offset,broad_max,max_amp,Hgamma_WL+max_offset,broad_max,
              max_amp,HeII_WL+max_offset,broad_max,
              max_amp,Hbeta_WL+max_offset,broad_max,max_amp,Hbeta_WL+max_offset,broad_max,
              max_amp,Hbeta_WL+max_offset,broad_max,max_amp,Hbeta_WL+max_offset,broad_max,
              max_amp,
              max_amp,OIII2_WL+max_offset,narrow_max,max_amp,OIII2_WL+max_offset,narrow_max]


        

####################################################################################################################################
############# SpecDecomp Class
####################################################################################################################################

class SpecDecomp():
    
    def __init__(self, wave, flux, err, feii_template_path, stellar_template_path, balmer_cont_template_path, balmer_highorder_template_path):
        '''
        Initializes spectrum, grabs FeII templates and stellar templates
        '''
        
        # prune bad values from wave,flux, and err
        self.wave_init=wave
        self.flux_init=flux
        self.err_init=err
        self.wave,self.flux,self.err=self.clip_data(wave,flux,err)
        
        self.feii_templates=np.load(feii_template_path,allow_pickle=True)
        self.stellar_templates=np.genfromtxt(stellar_template_path,skip_header=5)
        self.balmer_cont_templates=np.load(balmer_cont_template_path,allow_pickle=True)
        self.balmer_highorder_templates=np.load(balmer_highorder_template_path,allow_pickle=True)
        
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
    
    def emission_lines(self,wave,params,return_gaussians=False,param_dict=True):
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
           
        - if param_dict=False, params should be a list of only the emission parameters given below   
        - if param_dict=True, params should be a dictionary that includes the names
        
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
        params['ampOIII50071']
        params['WLOIII50071']
        params['widthOIII50071']
        params['ampOIII50072']
        params['WLOIII50072']
        params['widthOIII50072']
        '''
        if param_dict:
            params=[params['ampOII3729'],
                    params['ampOII3735'],
                    params['ampNeIII3868'],
                    params['ampNeIII3966'],
                    params['ampSII4073'],
                    params['groupampratio'],
                    params['groupWLoffset1'],
                    params['groupWLoffset2'],
                    params['groupwidth1'],
                    params['groupwidth2'],
                    params['ampHDelta'],params['WLHDelta'],params['widthHDelta'],
                    params['ampHGamma'],params['WLHGamma'],params['widthHGamma'],
                    params['ampHeII'],params['WLHeII'],params['widthHeII'],
                    params['ampHBeta1'],params['WLHBeta1'],params['widthHBeta1'],
                    params['ampHBeta2'],params['WLHBeta2'],params['widthHBeta2'],
                    params['ampHBeta3'],params['WLHBeta3'],params['widthHBeta3'],
                    params['ampHBeta4'],params['WLHBeta4'],params['widthHBeta4'],
                    params['ampNHBeta'],
                    params['ampOIII50071'],params['WLOIII50071'],params['widthOIII50071'],
                    params['ampOIII50072'],params['WLOIII50072'],params['widthOIII50072']]
        
        gaussians=[]
        
        # Construct gaussians for narrow lines near 4000 A
        WLs_group=[OII1_WL,OII2_WL,NeIII1_WL,NeIII2_WL,SII_WL]
        for i in range(5): # 5 lines in this complex
            amp_gauss1=params[i] # free parameter for amplitude of this line
            amp_gauss2=amp_gauss1*params[5] # amp gauss1 * group amp ratio
            WL_gauss1=WLs_group[i]+params[6] # rest WL + group offset 1
            WL_gauss2=WLs_group[i]+params[7] # rest WL + group offset 2
            width_gauss1=params[8] # group width 1
            width_gauss2=params[9] # group width 2
            
            gaussians.append(self.gaussian(wave,amp_gauss1,WL_gauss1,width_gauss1))
            gaussians.append(self.gaussian(wave,amp_gauss2,WL_gauss2,width_gauss2))
            
        # Construct H Delta, H Gamma, He II, and Broad H Beta gaussians
        for i in range(7): # 1 H Delta, 1 H Gamma, 1 He II, 4 Broad H Beta
            gaussians.append(self.gaussian(wave,params[10+(i*3)],params[11+(i*3)],params[12+(i*3)]))
            
        # Construct Narrow H Beta gaussians
        amp_gauss1=params[31] # amp N Hbeta1
        WL_gauss1=params[33]-(OIII2_WL-Hbeta_WL)  # WL OIII 5007 1 - offset
        width_gauss1=params[34] # width OIII 5007 1
        
        amp_gauss2=params[31]*params[35]/params[32]  # amp N Hbeta1 * amp OIII 5007 2 / amp OIII 5007 1
        WL_gauss2=params[36]-(OIII2_WL-Hbeta_WL) # WL OIII 5007 2 - offset
        width_gauss2=params[37] # width OIII 5007 2
        
        gaussians.append(self.gaussian(wave,amp_gauss1,WL_gauss1,width_gauss1))
        gaussians.append(self.gaussian(wave,amp_gauss2,WL_gauss2,width_gauss2))
        
        # Construct OIII 4960 gaussians
        amp_gauss1=(1/3)*params[32] # 1/3 amp OIII 5007 1
        WL_gauss1=params[33]-(OIII2_WL-OIII1_WL) # WL OIII 5007 1 - offset
        width_gauss1=params[34] # width OIII 5007 1
        
        amp_gauss2=(1/3)*params[35] # 1/3 amp OIII 5007 2
        WL_gauss2=params[36]-(OIII2_WL-OIII1_WL) # WL OIII 5007 2 - offset
        width_gauss2=params[37] # width OIII 5007 2
        
        gaussians.append(self.gaussian(wave,amp_gauss1,WL_gauss1,width_gauss1))
        gaussians.append(self.gaussian(wave,amp_gauss2,WL_gauss2,width_gauss2))
        
        # Construct OIII 5007 gaussians
        for i in range(2):
            gaussians.append(self.gaussian(wave,params[32+(i*3)],params[33+(i*3)],params[34+(i*3)]))
            
        if return_gaussians:
            return gaussians
            
        emission=np.zeros(len(wave))
        for i in gaussians:
            emission+=i
            
        return emission
            
    def power_law(self,wave,norm,slope):
        '''
        continuum power law. Normalized at a specified WL.
        
        params:
            wave - wavelengths of data
            norm - power-law normalization factor
            slope - power-law slope
        '''
        
        return norm*(wave/pl_norm_wl)**slope
    
    def construct_model(self,wave,params,fit_type):
        '''
        constructs the spectral decomposition model
        
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
        params[10-] - emission line parameters (check emission_lines() function)
        '''
        power_law=self.power_law(wave,params['pl_norm'],params['pl_alpha'])
        fe_ii_op=self.template_fitter(wave,self.feii_templates,params['fe_ii_width'],params['fe_ii_scale'])
        host=self.template_fitter(wave,self.stellar_templates,params['host_age'],params['host_scale'])
        balmer_cont=self.template_fitter(wave,self.balmer_cont_templates,params['bal_cont_tau'],params['bal_cont_scale'])
        balmer_highorder=self.template_fitter(wave,self.balmer_highorder_templates,params['bal_high_width'],params['bal_high_scale'])
        balmer=balmer_cont+balmer_highorder
        
        if fit_type=='cont':
            model=power_law+fe_ii_op+host+balmer
        elif fit_type=='full': 
            emission=self.emission_lines(wave,params)
            model=power_law+fe_ii_op+host+balmer+emission
        else:
            raise ValueError("Fit Type Must Be 'full' or 'cont', not '{:s}'".format(fit_type))

        return model
    
    def guess_initial_params(self):
        '''
        Uses the spectrum to guess initial values for model params
        '''
        
        # set power law normalization equal to flux at pl_norm_wl
        
        # first handle case where pl_norm_wl not in bandpass
        if np.amin(self.wave)>pl_norm_wl:
            # take first index
            idx_low=0
        else:
            for i in range(len(self.wave)):
                if self.wave[i]>=pl_norm_wl:
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
        alpha=(np.log(flux_5500)-np.log(pl_norm))/(np.log(5500/pl_norm_wl))
        if np.isnan(alpha):
            alpha=-1.8
        
        params_init={}
        params_init['pl_norm']=pl_norm-2
        params_init['pl_alpha']=alpha
        params_init['fe_ii_width']=30
        params_init['fe_ii_scale']=1
        params_init['host_age']=4
        params_init['host_scale']=1
        params_init['bal_cont_tau']=10
        params_init['bal_cont_scale']=1
        params_init['bal_high_width']=10
        params_init['bal_high_scale']=1
        
        # guess amplitudes of emission lines based on flux at center wavelength - power law
        pl_init=self.power_law(self.wave,pl_norm,alpha)
        
        # Initialize narrow lines around 4000 A with WL offsets of 0, narrow width, and amplitude of 1
        params_init['ampOII3729']=1
        params_init['ampOII3735']=1
        params_init['ampNeIII3868']=1
        params_init['ampNeIII3966']=1
        params_init['ampSII4073']=1
        params_init['groupampratio']=1
        params_init['groupWLoffset1']=0
        params_init['groupWLoffset2']=0
        params_init['groupwidth1']=narrow_start
        params_init['groupwidth2']=narrow_start
        
        # Guess amplitude of H Delta, H Gamma, H Beta, and OIII 5007
        # based on flux at center wavelength - power law at that wavelength
        
        # H Delta
        idx_HDelta=np.where(self.wave>Hdelta_WL)[0][0]
        amp_HDelta=self.flux[idx_HDelta]-pl_init[idx_HDelta]
        params_init['ampHDelta']=amp_HDelta
        params_init['WLHDelta']=Hdelta_WL
        params_init['widthHDelta']=broad_start
        
        # H Gamma
        idx_HGamma=np.where(self.wave>Hgamma_WL)[0][0]
        amp_HGamma=self.flux[idx_HGamma]-pl_init[idx_HGamma]
        params_init['ampHGamma']=amp_HGamma
        params_init['WLHGamma']=Hgamma_WL
        params_init['widthHGamma']=broad_start
        
        # He II
        idx_HeII=np.where(self.wave>HeII_WL)[0][0]
        amp_HeII=self.flux[idx_HeII]-pl_init[idx_HeII]
        params_init['ampHeII']=amp_HeII
        params_init['WLHeII']=HeII_WL
        params_init['widthHeII']=broad_start
        
        # Broad H Beta
        idx_HBeta=np.where(self.wave>Hbeta_WL)[0][0]
        amp_HBeta=self.flux[idx_HBeta]-pl_init[idx_HBeta]
        params_init['ampHBeta1']=amp_HBeta/4
        params_init['WLHBeta1']=Hbeta_WL-15
        params_init['widthHBeta1']=broad_start
        
        params_init['ampHBeta2']=amp_HBeta/4
        params_init['WLHBeta2']=Hbeta_WL-5
        params_init['widthHBeta2']=broad_start
        
        params_init['ampHBeta3']=amp_HBeta/4
        params_init['WLHBeta3']=Hbeta_WL+5
        params_init['widthHBeta3']=broad_start
        
        params_init['ampHBeta4']=amp_HBeta/4
        params_init['WLHBeta4']=Hbeta_WL+15
        params_init['widthHBeta4']=broad_start
        
        # Narrow H Beta
        params_init['ampNHBeta']=1
        
        #OIII 5007
        idx_OIII2=np.where(self.wave>OIII2_WL)[0][0]
        amp_OIII2=self.flux[idx_OIII2]-pl_init[idx_OIII2]
        params_init['ampOIII50071']=amp_OIII2/2
        params_init['WLOIII50071']=OIII2_WL
        params_init['widthOIII50071']=narrow_start
        params_init['ampOIII50072']=amp_OIII2/2
        params_init['WLOIII50072']=OIII2_WL
        params_init['widthOIII50072']=narrow_start
        
        return params_init
        
    def residuals(self,params,data,fit_type='full'):
        '''
        residuals function for kmpfit. Returns (flux-model)/err, where model is constructed from params.

        params - numpy array of parameter values
        data - tuple with (wave,flux,err)
        
        '''
        
        wave,flux,err=data

        model=self.construct_model(wave,params,fit_type)
        
        resid=(flux-model)/err
        #resid_nonan=np.where(resid==np.nan,10,resid)
        
        return resid
    
    
    def fit(self,params_init,lower_bounds,upper_bounds,minimize_method,verbose=False):
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
        indexes=self.wave_region_select(self.wave,pl_feii_wls)
        data_fit1=(self.wave[indexes],self.flux[indexes],self.err[indexes])
        
        fit_params=Parameters()
        # Power Law and Fe II can vary
        fit_params.add('pl_norm',value=params_init['pl_norm'],min=lower_bounds[0],max=upper_bounds[0],vary=True)
        fit_params.add('pl_alpha',value=params_init['pl_alpha'],min=lower_bounds[1],max=upper_bounds[1],vary=True)
        fit_params.add('fe_ii_width',value=params_init['fe_ii_width'],min=lower_bounds[2],max=upper_bounds[2],vary=True)
        fit_params.add('fe_ii_scale',value=params_init['fe_ii_scale'],min=lower_bounds[3],max=upper_bounds[3],vary=True)
        # Fix other continuum parameters
        fit_params.add('host_age',value=params_init['host_age'],min=lower_bounds[4],max=upper_bounds[4],vary=False)
        fit_params.add('host_scale',value=params_init['host_scale'],min=lower_bounds[5],max=upper_bounds[5],vary=False)
        fit_params.add('bal_cont_tau',value=params_init['bal_cont_tau'],min=lower_bounds[6],max=upper_bounds[6],vary=False)
        fit_params.add('bal_cont_scale',value=params_init['bal_cont_scale'],min=lower_bounds[7],max=upper_bounds[7],vary=False)
        fit_params.add('bal_high_width',value=params_init['bal_high_width'],min=lower_bounds[8],max=upper_bounds[8],vary=False)
        fit_params.add('bal_high_scale',value=params_init['bal_high_scale'],min=lower_bounds[9],max=upper_bounds[9],vary=False)
        
        #fit_params.pretty_print()
        
        fit1_result=minimize(self.residuals, 
                             fit_params, 
                             args=(data_fit1,),
                             kws={'fit_type': 'cont'},
                             nan_policy=nan_policy,
                             method=minimize_method,
                             max_nfev=None)
        
        fit_params=fit1_result.params
        self.params_fit1=fit1_result.params
        
        # Fit 2: Host and Narrow Lines around 4000 A, also Balmer (all other parameters fixed)
        indexes=self.wave_region_select(self.wave,host_narrow_wls)
        data_fit2=(self.wave[indexes],self.flux[indexes],self.err[indexes])
        
        # First update status of continuum parameters
        fit_params['pl_norm'].set(vary=False)
        fit_params['pl_alpha'].set(vary=False)
        fit_params['fe_ii_width'].set(vary=False)
        fit_params['fe_ii_scale'].set(vary=False)
        fit_params['host_age'].set(vary=True)
        fit_params['host_scale'].set(vary=True)
        fit_params['bal_cont_tau'].set(vary=True)
        fit_params['bal_cont_scale'].set(vary=True)
        fit_params['bal_high_width'].set(vary=True)
        fit_params['bal_high_scale'].set(vary=True)
        
        
        # Now add parameters for all emission lines
        # Narrow lines around 4000 A are allowed to vary
        
        fit_params.add('ampOII3729',    value=params_init['ampOII3729'],    min=lower_bounds[10],max=upper_bounds[10],vary=True)
        fit_params.add('ampOII3735',    value=params_init['ampOII3735'],    min=lower_bounds[11],max=upper_bounds[11],vary=True)
        fit_params.add('ampNeIII3868',  value=params_init['ampNeIII3868'],  min=lower_bounds[12],max=upper_bounds[12],vary=True)
        fit_params.add('ampNeIII3966',  value=params_init['ampNeIII3966'],  min=lower_bounds[13],max=upper_bounds[13],vary=True)
        fit_params.add('ampSII4073',    value=params_init['ampSII4073'],    min=lower_bounds[14],max=upper_bounds[14],vary=True)
        fit_params.add('groupampratio', value=params_init['groupampratio'], min=lower_bounds[15],max=upper_bounds[15],vary=True)
        fit_params.add('groupWLoffset1',value=params_init['groupWLoffset1'],min=lower_bounds[16],max=upper_bounds[16],vary=True)
        fit_params.add('groupWLoffset2',value=params_init['groupWLoffset2'],min=lower_bounds[17],max=upper_bounds[17],vary=True)
        fit_params.add('groupwidth1',   value=params_init['groupwidth1'],   min=lower_bounds[18],max=upper_bounds[18],vary=True)
        fit_params.add('groupwidth2',   value=params_init['groupwidth2'],   min=lower_bounds[19],max=upper_bounds[19],vary=True)
        
        # Other lines are fixed
        fit_params.add('ampHDelta',     value=params_init['ampHDelta'],     min=lower_bounds[20],max=upper_bounds[20],vary=False)
        fit_params.add('WLHDelta',      value=params_init['WLHDelta'],      min=lower_bounds[21],max=upper_bounds[21],vary=False)
        fit_params.add('widthHDelta',   value=params_init['widthHDelta'],   min=lower_bounds[22],max=upper_bounds[22],vary=False)
        fit_params.add('ampHGamma',     value=params_init['ampHGamma'],     min=lower_bounds[23],max=upper_bounds[23],vary=False)
        fit_params.add('WLHGamma',      value=params_init['WLHGamma'],      min=lower_bounds[24],max=upper_bounds[24],vary=False)
        fit_params.add('widthHGamma',   value=params_init['widthHGamma'],   min=lower_bounds[25],max=upper_bounds[25],vary=False)
        
        fit_params.add('ampHeII',       value=params_init['ampHeII'],       min=lower_bounds[26],max=upper_bounds[26],vary=False)
        fit_params.add('WLHeII',        value=params_init['WLHeII'],        min=lower_bounds[27],max=upper_bounds[27],vary=False)
        fit_params.add('widthHeII',     value=params_init['widthHeII'],     min=lower_bounds[28],max=upper_bounds[28],vary=False)
        
        fit_params.add('ampHBeta1',     value=params_init['ampHBeta1'],     min=lower_bounds[26],max=upper_bounds[29],vary=False)
        fit_params.add('WLHBeta1',      value=params_init['WLHBeta1'],      min=lower_bounds[27],max=upper_bounds[30],vary=False)
        fit_params.add('widthHBeta1',   value=params_init['widthHBeta1'],   min=lower_bounds[28],max=upper_bounds[31],vary=False)
        fit_params.add('ampHBeta2',     value=params_init['ampHBeta2'],     min=lower_bounds[29],max=upper_bounds[32],vary=False)
        fit_params.add('WLHBeta2',      value=params_init['WLHBeta2'],      min=lower_bounds[30],max=upper_bounds[33],vary=False)
        fit_params.add('widthHBeta2',   value=params_init['widthHBeta2'],   min=lower_bounds[31],max=upper_bounds[34],vary=False)
        fit_params.add('ampHBeta3',     value=params_init['ampHBeta3'],     min=lower_bounds[32],max=upper_bounds[35],vary=False)
        fit_params.add('WLHBeta3',      value=params_init['WLHBeta3'],      min=lower_bounds[33],max=upper_bounds[36],vary=False)
        fit_params.add('widthHBeta3',   value=params_init['widthHBeta3'],   min=lower_bounds[34],max=upper_bounds[37],vary=False)
        fit_params.add('ampHBeta4',     value=params_init['ampHBeta4'],     min=lower_bounds[35],max=upper_bounds[38],vary=False)
        fit_params.add('WLHBeta4',      value=params_init['WLHBeta4'],      min=lower_bounds[36],max=upper_bounds[39],vary=False)
        fit_params.add('widthHBeta4',   value=params_init['widthHBeta4'],   min=lower_bounds[37],max=upper_bounds[40],vary=False)
        fit_params.add('ampNHBeta',     value=params_init['ampNHBeta'],     min=lower_bounds[38],max=upper_bounds[41],vary=False)
        fit_params.add('ampOIII50071',  value=params_init['ampOIII50071'],  min=lower_bounds[39],max=upper_bounds[42],vary=False)
        fit_params.add('WLOIII50071',   value=params_init['WLOIII50071'],   min=lower_bounds[40],max=upper_bounds[43],vary=False)
        fit_params.add('widthOIII50071',value=params_init['widthOIII50071'],min=lower_bounds[41],max=upper_bounds[44],vary=False)
        fit_params.add('ampOIII50072',  value=params_init['ampOIII50072'],  min=lower_bounds[42],max=upper_bounds[45],vary=False)
        fit_params.add('WLOIII50072',   value=params_init['WLOIII50072'],   min=lower_bounds[43],max=upper_bounds[46],vary=False)
        fit_params.add('widthOIII50072',value=params_init['widthOIII50072'],min=lower_bounds[44],max=upper_bounds[47],vary=False)
        
        fit2_result=minimize(self.residuals, 
                             fit_params, 
                             args=(data_fit2,),
                             kws={'fit_type': 'full'},
                             nan_policy=nan_policy,
                             method=minimize_method,
                             max_nfev=None)
        fit_params=fit2_result.params
        self.params_fit2=fit2_result.params
        
        # Fit 3: H Delta and H Gamma, all other params fixed
        indexes=self.wave_region_select(self.wave,hdelta_hgamma_wls)
        data_fit3=(self.wave[indexes],self.flux[indexes],self.err[indexes])
        
        # First set all params to not vary
        param_names=fit_params.valuesdict().keys()
        for i in param_names: 
            fit_params[i].set(vary=False)
        
        # Then set proper params to vary
        fit_params['ampHDelta'].set(vary=True)
        fit_params['WLHDelta'].set(vary=True)
        fit_params['widthHDelta'].set(vary=True)
        fit_params['ampHGamma'].set(vary=True)
        fit_params['WLHGamma'].set(vary=True)
        fit_params['widthHGamma'].set(vary=True)
        
        fit3_result=minimize(self.residuals, 
                             fit_params, 
                             args=(data_fit3,),
                             kws={'fit_type': 'full'},
                             nan_policy=nan_policy,
                             method=minimize_method,
                             max_nfev=None)
        
        fit_params=fit3_result.params
        self.params_fit3=fit3_result.params
        
        # Fit 4: Broad H Beta, Narrow H Beta, OIII 4960, OIII 5007, all other params fixed
        indexes=self.wave_region_select(self.wave,hbeta_oiii_wls)
        data_fit4=(self.wave[indexes],self.flux[indexes],self.err[indexes])
        
        # First set all params to not vary
        param_names=fit_params.valuesdict().keys()
        for i in param_names: 
            fit_params[i].set(vary=False)
            
        # Then set proper params to vary
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
        fit_params['ampOIII50071'].set(vary=True)
        fit_params['WLOIII50071'].set(vary=True)
        fit_params['widthOIII50071'].set(vary=True)
        fit_params['ampOIII50072'].set(vary=True)
        fit_params['WLOIII50072'].set(vary=True)
        fit_params['widthOIII50072'].set(vary=True)
        
        fit4_result=minimize(self.residuals, 
                             fit_params, 
                             args=(data_fit4,),
                             kws={'fit_type': 'full'},
                             nan_policy=nan_policy,
                             method=minimize_method,
                             max_nfev=None)
        
        fit_params=fit4_result.params
        self.params_fit4=fit4_result.params
        
        # Fit 5: All parameters can vary
        data_fit5=(self.wave,self.flux,self.err)
        
        param_names=fit_params.valuesdict().keys()
        for i in param_names: 
            fit_params[i].set(vary=True)
            
        #self.params_fit4.pretty_print()
            
        result=minimize(self.residuals, 
                        fit_params, 
                        args=(data_fit5,),
                        kws={'fit_type': 'full'},
                        nan_policy=nan_policy,
                        method=minimize_method,
                        max_nfev=None)
        
        fit_params=result.params
        
        self.params_final=result.params
        self.results_final=result
        self.chisq=result.chisqr
        self.red_chisq=result.redchi
        self.aic=result.aic
        self.bic=result.bic
        
        self.p_law=self.power_law(self.wave,self.params_final['pl_norm'],self.params_final['pl_alpha'])
        self.fe_ii=self.template_fitter(self.wave,self.feii_templates,self.params_final['fe_ii_width'],self.params_final['fe_ii_scale'])
        self.host=self.template_fitter(self.wave,self.stellar_templates,self.params_final['host_age'],self.params_final['host_scale'])
        bal_cont=self.template_fitter(self.wave,self.balmer_cont_templates,self.params_final['bal_cont_tau'],self.params_final['bal_cont_scale'])
        bal_high=self.template_fitter(self.wave,self.balmer_highorder_templates,self.params_final['bal_high_width'],self.params_final['bal_high_scale'])
        self.balmer=bal_cont+bal_high
        self.emission=self.emission_lines(self.wave,self.params_final)
        
        time1=time.time()
        
        if verbose:
            
            print('Fitting Time: {:.1f} s'.format(time1-time0))
        
        return self.params_final
    
    def construct_model_emcee(self,params):
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
        params[10-] - emission line parameters 
        '''
        power_law=self.power_law(self.wave,params[0],params[1])
        fe_ii_op=self.template_fitter(self.wave,self.feii_templates,params[2],params[3])
        host=self.template_fitter(self.wave,self.stellar_templates,params[4],params[5])
        balmer_cont=self.template_fitter(self.wave,self.balmer_cont_templates,params[6],params[7])
        balmer_highorder=self.template_fitter(self.wave,self.balmer_highorder_templates,params[8],params[9])
        balmer=balmer_cont+balmer_highorder
        emission=self.emission_lines(self.wave,params[10:],return_gaussians=False,param_dict=False)
        
        model=power_law+fe_ii_op+host+balmer+emission
        
        return model
   
    def log_likelihood(self,params):
        '''
        log likelihood estimation for emcee
        '''
        
        model=self.construct_model_emcee(params)

        
        return -0.5 * np.sum((self.flux - model) ** 2 / self.err**2)
    
    def log_prior(self,params):
        '''
        uniform priors for emcee
        '''
        
        for i in range(len(params)):
            
            if params[i]<self.lower_bounds[i]:
                return -np.inf
            if params[i]>self.upper_bounds[i]:
                return -np.inf
        return 0.0
    
    
    def fit_emcee(self,params_init,lower_bounds,upper_bounds,nwalkers,nsteps):
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
        
        def log_probability(params):
            '''
            posterior for emcee
            '''

            lp=self.log_prior(params)
            if not np.isfinite(lp):
                return -np.inf
            return lp + self.log_likelihood(params)
        
        
        self.lower_bounds=lower_bounds
        self.upper_bounds=upper_bounds
        params_list=list([value for key,value in params_init.items()])
        ndim=len(params_list)
        
        # Initialize walkers in a small Gaussian ball around initial parameter values
        pos=params_list+ 1e-4 * np.random.randn(nwalkers, ndim)
        
        # Run Emcee
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
        sampler.run_mcmc(pos, nsteps,progress=True)
        
        self.sampler=sampler
        
        return self.sampler
        
        



   
