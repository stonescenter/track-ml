import pandas as pd
import sys

from math import sqrt
import sys
import scipy.stats
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

sys.path.append('/home/silvio/git/track-ml-1/utils')
from tracktop import *


#def create_graphic(reconstructed_tracks, original_tracks, tracks_diffs):
def create_graphic_org(**kwargs):

    if kwargs.get('original_tracks'):
        original_tracks = kwargs.get('original_tracks')

    if kwargs.get('path_original_track'):
        path_original_track = kwargs.get('path_original_track')

    if kwargs.get('tracks'):
        tracks = kwargs.get('tracks')

    dfOriginal = pd.read_csv(original_tracks)
#    dfOriginal2=dfOriginal.iloc(10:,:)

    track_plot_new(dfOriginal,  track_color = 'blue', n_tracks = tracks, title = 'Original to be Reconstructed', path=path_original_track)

#def create_graphic(reconstructed_tracks, original_tracks, tracks_diffs):
def create_graphic(**kwargs):

    if kwargs.get('reconstructed_tracks'):
        reconstructed_tracks = kwargs.get('reconstructed_tracks')

    if kwargs.get('original_tracks'):
        original_tracks = kwargs.get('original_tracks')

    if kwargs.get('tracks_diffs'):
        tracks_diffs = kwargs.get('tracks_diffs')

    if kwargs.get('path_original_track'):
        path_original_track = kwargs.get('path_original_track')

    if kwargs.get('path_recons_track'):
        path_recons_track = kwargs.get('path_recons_track')

    '''
    dftracks_diffs_aux      = pd.read_csv(tracks_diffs)
    dftracks_diffs=dftracks_diffs_aux.iloc[:,28:29]

    dfOriginal      = pd.read_csv(original_tracks)
    dfaux3=dfOriginal.drop(dfOriginal.columns[0], axis=1)
    dfaux32=dfaux3.drop(dfaux3.columns[0], axis=1)
    dfaux32.insert(174, "diff", dftracks_diffs, True)
    org = dfaux32.sort_values(by=['diff'])

    dfRecons      = pd.read_csv(reconstructed_tracks)
    dfaux33=dfRecons.drop(dfRecons.columns[0], axis=1)
    dfaux22=dfaux33.drop(dfaux33.columns[0], axis=1)
    dfaux22.insert(65, "diff", dftracks_diffs, True)
    org2 = dfaux22.sort_values(by=['diff'])
    '''

    dfRecons   = pd.read_csv(reconstructed_tracks)
    dfOriginal = pd.read_csv(original_tracks)

    dftracks_diffs_aux   = pd.read_csv(tracks_diffs)
    dftracks_diffs=dftracks_diffs_aux.iloc[:,28:29]

    dfRecons.insert(171, "diff", dftracks_diffs, True)
    dfOriginal.insert(171, "diff", dftracks_diffs, True)

    recaux=dfRecons.iloc[:,12:]
    orgaux=dfOriginal.iloc[:,11:]

    org = orgaux.sort_values(by=['diff'])
    rec = recaux.sort_values(by=['diff'])

    track_plot_new(org,  track_color = 'blue', n_tracks = 19, title = 'Original to be Reconstructed', path=path_original_track)
    #track_plot(org,  track_color = 'blue', n_tracks = 20, title = 'Original to be Reconstructed', path=path_recons_track)
    track_plot_new(rec, track_color = 'red', n_tracks = 19, title = 'reconstructed LSTM', path=path_recons_track)
    #track_plot(org2, track_color = 'red', n_tracks = 20, title = 'reconstructed LSTM', path=path_original_track)

#def create_histogram(tracks_diffs):
def create_histogram(**kwargs):
    if kwargs.get('tracks_diffs'):
        tracks_diffs = kwargs.get('tracks_diffs')

    if kwargs.get('path_hist'):
        path_hist = kwargs.get('path_hist')

    dftracks_diffs_aux     = pd.read_csv(tracks_diffs)
    dftracks_diffs         = dftracks_diffs_aux.iloc[:,28:29]
    dftracks_diffs_aux[29] = (dftracks_diffs_aux.iloc[:,28:29]/10)
    dfff=dftracks_diffs_aux.sort_values(by=[29])

    track_plot_hist(dfff.iloc[:,-1:], title = "AVG distance - Real x LSTM",
                    x_title = "Average of Distance (cm)",
                    y_title = "frequency",
                    n_bins = 20, bar_color = 'indianred',
                    path = path_hist)

#def create_histogram_seaborn(tracks_diffs,outputfig):
def create_histogram_seaborn(**kwargs):
    if kwargs.get('tracks_diffs'):
        tracks_diffs = kwargs.get('tracks_diffs')
    if kwargs.get('outputfig'):
        outputfig = kwargs.get('outputfig')

    dftracks_diffs_aux     = pd.read_csv(tracks_diffs)
    dftracks_diffs         = dftracks_diffs_aux.iloc[:,28:29]
    dftracks_diffs_aux[29] = (dftracks_diffs_aux.iloc[:,28:29]/10)
    dfff=dftracks_diffs_aux.sort_values(by=[29])

    #sns_plot = sns.distplot(dfEval.iloc[:,[27]])
    sns_plot = sns.distplot(dfff.iloc[:,-1:])

    sns_plot.set(xlabel='Average Distance in MM', ylabel='Frequency')
    plt.savefig(outputfig)


#def create_diference_per_track(reconstructed_tracks, original_tracks, eval_file):
def create_diference_per_track(**kwargs):

    if kwargs.get('reconstructed_tracks'):
        reconstructed_tracks = kwargs.get('reconstructed_tracks')

    if kwargs.get('original_tracks'):
        original_tracks = kwargs.get('original_tracks')

    if kwargs.get('eval_file'):
        eval_file = kwargs.get('eval_file')

    dfOriginal      = pd.read_csv(original_tracks)
    dfReconstructed = pd.read_csv(reconstructed_tracks)

    lines   = dfOriginal.shape[0]
    columns = dfOriginal.shape[1]

    dfEval  = pd.DataFrame(index=range(lines),columns=range(29))

    ind_dfEval=0

    #for hit in range(1, 28):
    for hit in range(1, 16):
        print(hit)
        print(dfOriginal.shape)
        print(dfReconstructed.shape)
        #original track
        dataOR=dfOriginal.iloc[:, [ (hit*8)+2,(hit*8)+3,(hit*8)+4 ]]
        #reconstructed track
        dataRE=dfReconstructed.iloc[:, [ (hit*8)+2,(hit*8)+3,(hit*8)+4 ]]

        dftemp = pd.DataFrame(index=range(lines),columns=range(7))
        dftemp[0]=dataOR.iloc[:,[0]]
        dftemp[1]=dataOR.iloc[:,[1]]
        dftemp[2]=dataOR.iloc[:,[2]]

        dftemp[3]=dataRE.iloc[:,[0]]
        dftemp[4]=dataRE.iloc[:,[1]]
        dftemp[5]=dataRE.iloc[:,[2]]

        dftemp[6]=   (((dftemp[0]-dftemp[3])**2)+((dftemp[1]-dftemp[4])**2)+((dftemp[2]-dftemp[5])**2)).pow(1./2)

        #Dataframe with geometric distante from hit to hit
        dfEval[ind_dfEval] = dftemp[6]
        ind_dfEval=ind_dfEval+1

    ind=27
    col = dfEval.loc[: , 0:26]
    dfEval[27] = col.mean(axis=1)
    dfEval.to_csv(eval_file)


def create_input_data(**kwargs):
    #this function select a specific number of track with specific amount of hists
    maximunAmountofHitsinDB=20
    columnsperhit=8
    firstColumnAfterParticle=9

    if kwargs.get('event_prefix'):
        event_prefix = kwargs.get('event_prefix')

    if kwargs.get('output_prefix'):
        output_prefix = kwargs.get('output_prefix')

    if kwargs.get('aux_am_per_hit'):
        aux_am_per_hit = kwargs.get('aux_am_per_hit')

    if kwargs.get('min'):
        min = kwargs.get('min')

    if kwargs.get('max'):
        max = kwargs.get('max')

    if kwargs.get('maximunAmountofHitsinDB'):
        maximunAmountofHitsinDB = kwargs.get('maximunAmountofHitsinDB')

    if kwargs.get('columnsperhit'):
        columnsperhit = kwargs.get('columnsperhit')

    if kwargs.get('firstColumnAfterParticle'):
        firstColumnAfterParticle = kwargs.get('firstColumnAfterParticle')

    nrowsvar=1500000 #1000 #500000
    skip=0
    totdf = pd.read_csv(event_prefix,skiprows=0,nrows=1)
    totdf = totdf.iloc[0:0]

    for z in range(1):

        dfOriginal = pd.read_csv(event_prefix,skiprows=skip,nrows=nrowsvar)
        #print(dfOriginal.shape)
        #dfOriginal2=dfOriginal.iloc[:,7:]
        dfOriginal2=dfOriginal.iloc[:,firstColumnAfterParticle:]
        #print(dfOriginal2)

        #all zero columns in a single line
        dfOriginal['totalZeros'] = (dfOriginal2 == 0.0).sum(axis=1)
        dfOriginal['totalHits'] = maximunAmountofHitsinDB-(dfOriginal['totalZeros']/columnsperhit)

        dfOriginal["totalHits"] = dfOriginal["totalHits"].astype(int)

        #print("min: " , dfOriginal.iloc[:,:-1].min())
        #print("max: " , dfOriginal.iloc[:,:-1].max())

        print(int(min))
        print(int(max)+1)
        #print(dfOriginal['totalZeros'])
        #print(dfOriginal['totalHits'])
        for i in range(int(min), (int(max)+1)):
            #print("i: " , i)
            auxDF = dfOriginal.loc[dfOriginal['totalHits'] == i]
            auxDF2 = auxDF.iloc[0:aux_am_per_hit,:]
            totdf = totdf.append(auxDF2, sort=False)
            print("auxDF2.shape: ", i, auxDF2.shape)

        dfOriginal = dfOriginal.iloc[0:0]
        auxDF2 = auxDF2.iloc[0:0]
        auxDF = auxDF.iloc[0:0]

    skip=skip+nrowsvar

    totdf.drop('totalHits', axis=1, inplace=True)
    totdf.drop('totalZeros', axis=1, inplace=True)
    totdf.to_csv(output_prefix, index = False)

def put_each_hit_in_a_single_line_train(**kwargs):

    if kwargs.get('event_prefix'):
        event_prefix = kwargs.get('event_prefix')

    if kwargs.get('output_prefix'):
        output_prefix = kwargs.get('output_prefix')


    print("event_prefix ", event_prefix, " output_prefix ", output_prefix)
    auxdf3  = pd.DataFrame()
    totfinaldfaux = pd.DataFrame()

    totdf = pd.read_csv(event_prefix) #,skiprows=0,nrows=1)
    print("totdf")
    print(totdf)
    print("totdf par")
    print(totdf.iloc[:,0:10])
    print("totdf hit 1")
    print(totdf.iloc[:,9:18])
    print("totdf hit 2")
    print(totdf.iloc[:,18:26])
    print("totdf hit 3")
    print(totdf.iloc[:,26:34])
    print("totdf hit 4")
    print(totdf.iloc[:,34:42])
    print("totdf hit 5")
    print(totdf.iloc[:,42:50])


    #print(totdf.shape)
    #totdf.drop('Unnamed: 0', axis=1, inplace=True)
    print(totdf.shape)

    columnsPerHit=8
    maximumHitPerTrack=20
    positionB=1
    #positionE=40 #17+8+8+8+8
    positionE=49 #17+8+8+8+8
    amount_of_hits = maximumHitPerTrack*columnsPerHit

    #define interval of first hit as the third hit
    #positionB=(((positionB+columnsPerHit)+columnsPerHit)+columnsPerHit)
    #positionE=(((positionE+columnsPerHit)+columnsPerHit)+columnsPerHit)

    #initialize dataframe that will receive each hit as a row
    #totfinaldfaux = pd.DataFrame(columns=range(columnsPerHit))

    #for from 3 to 20 -> 17 hits
    #for k in range(3,int(maximumHitPerTrack)-3):
    for k in range(3,4):
    #for k in range(3,10):
        print("K: ", k)
        print("interval: ", positionB,positionE)

        totdf3 = totdf.iloc[:,positionB:positionE]
        print("totdf3.shape: ", totdf3.shape)
        print("totdf3: ", totdf3)

        #rename column names to append in resulta dataframe
        for i in range(totdf3.columns.shape[0]):
            totdf3 = totdf3.rename(columns={totdf3.columns[i]: i})

        '''
        totdf3['totalZeros'] = (totdf3.iloc[:,:] == 0.0).sum(axis=1)

        #print( totdf3['totalZeros'] )
        print( "totdf3: " , totdf3.shape )
        auxdf3 = totdf3[ totdf3['totalZeros'] == 32 ]
        print( "auxdf3 :" , auxdf3.shape  )

        # Get names of indexes for which column Age has value 30
        indexNames = totdf3[ totdf3['totalZeros'] == 32 ].index

        # Delete these row indexes from dataFrame
        totdf3.drop(indexNames , inplace=True)
        print( "totdf3 after drop :" , totdf3.shape  )
        '''

        #print("totalZeros!!")
        #auxdf3['totalZeros'] = (totdf3 == 0.0).sum(axis=1)

        #for auxIndex, auxRow in totdf3.iterrows():
            #obj=(totdf3.iloc[auxIndex:auxIndex+1,:] == 0.0).sum(axis=1)

            #obj=obj.flatten()
            #print(obj)
            #print(obj.shape)
            #print(obj.iloc[0:10])
            #print(obj[1])
            #if (((totdf3.iloc[auxIndex:auxIndex+1,:] == 0.0).sum(axis=1)) > 32):
            #    print("dropRow")
            #else:
            #   print("keepRow")
            #df3d.drop('X', axis=1, inplace=True)
            #totdf3 = totdf3.rename(columns={totdf3.columns[i]: i})

        #print(auxdf3)
        #append hit as row
        totfinaldfaux = totfinaldfaux.append(totdf3)

        positionB=positionB+columnsPerHit
        positionE=positionE+columnsPerHit

    print(totfinaldfaux.shape)
    print(totfinaldfaux)
    totfinaldfaux.to_csv(output_prefix)
    #totfinaldfaux.to_csv(output_prefix)

'''
def put_each_hit_in_a_single_line_v2(**kwargs):
    if kwargs.get('event_prefix'):
        event_prefix = kwargs.get('event_prefix')

    if kwargs.get('output_prefix'):
        output_prefix = kwargs.get('output_prefix')

    totdf = pd.read_csv(event_prefix) #,skiprows=0,nrows=1)

    resorg= totdf.iloc[:, [ 26,27,28,29,30,31,32]]
    totfinaldfaux.to_csv(output_prefix)
'''

def put_each_hit_in_a_single_line(**kwargs):

    if kwargs.get('event_prefix'):
        event_prefix = kwargs.get('event_prefix')

    if kwargs.get('output_prefix'):
        output_prefix = kwargs.get('output_prefix')

    totdf = pd.read_csv(event_prefix) #,skiprows=0,nrows=1)
    print(totdf)
    print(totdf.shape)

    columnsPerHit=8
    maximumHitPerTrack=20
    #positionB=10
    #positionE=18
    positionB=1
    positionE=8
    amount_of_hits = maximumHitPerTrack*columnsPerHit

    #define interval of first hit as the third hit
    positionB=(((positionB+columnsPerHit)+columnsPerHit)+columnsPerHit)
    positionE=(((positionE+columnsPerHit)+columnsPerHit)+columnsPerHit)

    #initialize dataframe that will receive each hit as a row
    totfinaldfaux = pd.DataFrame(columns=range(columnsPerHit))

    #for from 3 to 20 -> 17 hits
    for k in range(3,int(maximumHitPerTrack)):
    #for k in range(1,1):

        print("interval: ", positionB,positionE)

        totdf3 = totdf.iloc[:,positionB:positionE]
        print(totdf3)

        #rename column names to append in resulta dataframe
        for i in range(totdf3.columns.shape[0]):
            totdf3 = totdf3.rename(columns={totdf3.columns[i]: i})

        #append hit as row
        totfinaldfaux = totfinaldfaux.append(totdf3)

        positionB=positionB+columnsPerHit
        positionE=positionE+columnsPerHit

    print(totfinaldfaux.shape)
    totfinaldfaux.to_csv(output_prefix)
    totfinaldfaux.to_csv(output_prefix)

def create_input_data_24(**kwargs):

    if kwargs.get('event_prefix'):
        event_prefix = kwargs.get('event_prefix')

    if kwargs.get('output_prefix'):
        output_prefix = kwargs.get('output_prefix')

    hitB = 2
    amount_of_hits = 24

    totdf = pd.read_csv(event_prefix) #,skiprows=0,nrows=1)

    totfinaldfaux = pd.DataFrame(columns=range(amount_of_hits))
    totfinaldfaux2 = pd.DataFrame(columns=range(amount_of_hits))

    for z in range(1,26):
        hitB = hitB+6
        hitEnd = hitB+amount_of_hits
        totfinaldf=totdf.iloc[:,hitB:hitEnd]

        totfinaldfaux2 = pd.DataFrame({0:totfinaldf.iloc[:,0], 1:totfinaldf.iloc[:,1], 2:totfinaldf.iloc[:,2], 3:totfinaldf.iloc[:,3], 4:totfinaldf.iloc[:,4], 5:totfinaldf.iloc[:,5], 6:totfinaldf.iloc[:,6], 7:totfinaldf.iloc[:,7] , 8:totfinaldf.iloc[:,8] , 9:totfinaldf.iloc[:,9], 10:totfinaldf.iloc[:,10], 11:totfinaldf.iloc[:,11], 12:totfinaldf.iloc[:,12], 13:totfinaldf.iloc[:,13], 14:totfinaldf.iloc[:,14], 15:totfinaldf.iloc[:,15], 16:totfinaldf.iloc[:,16], 17:totfinaldf.iloc[:,17] , 18:totfinaldf.iloc[:,18] , 19:totfinaldf.iloc[:,19], 20:totfinaldf.iloc[:,20], 21:totfinaldf.iloc[:,21], 22:totfinaldf.iloc[:,22] , 23:totfinaldf.iloc[:,23]  })

        totfinaldfaux = totfinaldfaux.append(totfinaldfaux2, sort=False, ignore_index=True)

    totfinaldfaux.to_csv(output_prefix)


def create_input_data_6(**kwargs):

    if kwargs.get('event_prefix'):
        event_prefix = kwargs.get('event_prefix')

    if kwargs.get('output_prefix'):
        output_prefix = kwargs.get('output_prefix')

    totdf = pd.read_csv(event_prefix) #,skiprows=0,nrows=1)

    hitB = 2
    amount_of_hits = 6
    #dfEval  = pd.DataFrame(index=range(lines),columns=range(29))

    totfinaldfaux = pd.DataFrame(columns=range(amount_of_hits))
    totfinaldfaux2 = pd.DataFrame(columns=range(amount_of_hits))

    #print(totfinaldfaux)
    ##totfinaldfaux = pd.DataFrame(columns = [0,1,2,3,4,5])
    #totfinaldfaux2 = pd.DataFrame(columns = [0,1,2,3,4,5])
    #print(totfinaldfaux2)

    #for z in range(1,26):
    for z in range(1,3):
        hitB = hitB+amount_of_hits
        hitEnd = hitB+amount_of_hits
        totfinaldf=totdf.iloc[:,hitB:hitEnd]
        print(hitB,hitEnd)

        totfinaldfaux2 = pd.DataFrame({0:totfinaldf.iloc[:,0], 1:totfinaldf.iloc[:,1], 2:totfinaldf.iloc[:,2], 3:totfinaldf.iloc[:,3], 4:totfinaldf.iloc[:,4], 5:totfinaldf.iloc[:,5]  })

        totfinaldfaux = totfinaldfaux.append(totfinaldfaux2, sort=False, ignore_index=True)

    print('to_csv')
    totfinaldfaux.to_csv(output_prefix)

def create_input_data_cfg(**kwargs):

    columnsPerHit=225
    maximumHitPerTrack=20
    if kwargs.get('event_prefix'):
        event_prefix = kwargs.get('event_prefix')

    if kwargs.get('output_prefix'):
        output_prefix = kwargs.get('output_prefix')

    if kwargs.get('columnsPerHit'):
        columnsPerHit = kwargs.get('columnsPerHit')

    if kwargs.get('maximumHitPerTrack'):
        maximumHitPerTrack = kwargs.get('maximumHitPerTrack')

    hitB = 2
    amount_of_hits = maximumHitPerTrack*columnsPerHit

    totdf = pd.read_csv(event_prefix) #,skiprows=0,nrows=1)

    totfinaldfaux = pd.DataFrame(columns=range(amount_of_hits))
    totfinaldfaux2 = pd.DataFrame(columns=range(amount_of_hits))

    for z in range(1,(maximumHitPerTrack-2)):
        hitB = hitB+columnsPerHit
        hitEnd = hitB+columnsPerHit*4
        totfinaldf=totdf.iloc[:,hitB:hitEnd]

        totfinaldfaux2 = pd.DataFrame({0:totfinaldf.iloc[:,0]})
        totfinaldfaux2 = pd.DataFrame({1:totfinaldf.iloc[:,1]})
        #totfinaldfaux2 = pd.DataFrame({0:totfinaldf.iloc[:,0], 1:totfinaldf.iloc[:,1], 2:totfinaldf.iloc[:,2], 3:totfinaldf.iloc[:,3], 4:totfinaldf.iloc[:,4], 5:totfinaldf.iloc[:,5], 6:totfinaldf.iloc[:,6], 7:totfinaldf.iloc[:,7] , 8:totfinaldf.iloc[:,8] , 9:totfinaldf.iloc[:,9], 10:totfinaldf.iloc[:,10], 11:totfinaldf.iloc[:,11], 12:totfinaldf.iloc[:,12], 13:totfinaldf.iloc[:,13], 14:totfinaldf.iloc[:,14], 15:totfinaldf.iloc[:,15], 16:totfinaldf.iloc[:,16], 17:totfinaldf.iloc[:,17] , 18:totfinaldf.iloc[:,18] , 19:totfinaldf.iloc[:,19], 20:totfinaldf.iloc[:,20], 21:totfinaldf.iloc[:,21], 22:totfinaldf.iloc[:,22] , 23:totfinaldf.iloc[:,23]  })


        totfinaldfaux = totfinaldfaux.append(totfinaldfaux2, sort=False, ignore_index=True)

    totfinaldfaux.to_csv(output_prefix)

#event_prefix         = sys.argv[1]
#output_prefix        = sys.argv[2]
#aux_am_per_hit       = int(sys.argv[3]) #250
#min                  = int(sys.argv[4]) #250
#max                  = int(sys.argv[5]) #250
#original_tracks      = sys.argv[6]
#reconstructed_tracks = sys.argv[7]
#outputfig            = sys.argv[8]
#eval_file            = sys.argv[9]

#create_input_data(event_prefix,output_prefix,aux_am_per_hit,min,max)
#create_input_data_6(event_prefix,output_prefix)
#create_input_data_24(event_prefix,output_prefix)

#create_diference_per_track(reconstructed_tracks, original_tracks, eval_file)
#create_graphic(reconstructed_tracks, original_tracks, eval_file)
#create_histogram(eval_file)
#create_histogram_seaborn(eval_file)
