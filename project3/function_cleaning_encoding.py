

feat_info_n = copy.deepcopy(feat_info)
feat_info_n=feat_info_n.drop(['information_level','type'],axis=1) #created new data frame to be able to pla with cleaning
feat_info_n['missing_or_unknown_']=feat_info_n['missing_or_unknown']
# get the splitted encode
for s in range(np.shape(feat_info_n)[0]):
    #print(len(str(feat_info_n['missing_or_unknown'][s])))
    if(len(str(feat_info_n['missing_or_unknown'][s]))==2):   #no clean out needed is 2 character []
        feat_info_n['missing_or_unknown_'][s]='pass'
        #print("['pass']")
    else:
        
        snew=str(feat_info_n['missing_or_unknown'][s]).replace('[','').replace(']','').split(',')    #get the values in real list
        #print(snew)
        feat_info_n['missing_or_unknown_'][s]=snew

feat_info_n=feat_info_n.drop(['missing_or_unknown'],axis=1) #drop extra col

azdias_ = copy.deepcopy(azdias)

print('start converting to nan')
for i,column_ in enumerate(azdias_.columns):
    if(i<86):
        missing_attr=feat_info_n[feat_info_n['attribute']==column_]['missing_or_unknown_'].values[0]
        if(missing_attr!='pass'):
            for j in range(len(missing_attr)):
                try:
                    azdias_.loc[azdias_[column_]==int(missing_attr[j]),column_]=np.nan
                except:
                    print(column_)
                    azdias_.loc[azdias_[column_]=='X',column_]=np.nan
                    azdias_.loc[azdias_[column_]=='XX',column_]=np.nan

print('finish converting to nan')
#drop the columns with outlier data that was evident from original data
azdias_out_remove = copy.deepcopy(azdias_)
print('drop outliers col')
azdias_out_remove=azdias_out_remove.drop(['AGER_TYP','GEBURTSJAHR','TITEL_KZ','ALTER_HH','KK_KUNDENTYP','KBA05_BAUMAX'],axis=1)

print('drop outliers row')
azdias_no_miss=azdias_out_remove.dropna()

print('write no miss data to file')
azdias_no_miss.to_csv('customer_no_miss_data.csv',index=False) 

#get useful numeric columns
print('make separate df of numeric col only')
azdias_no_miss_num_keep=pd.DataFrame(azdias_no_miss[['ALTERSKATEGORIE_GROB', 'FINANZ_MINIMALIST', 'FINANZ_SPARER',
       'FINANZ_VORSORGER', 'FINANZ_ANLEGER', 'FINANZ_UNAUFFAELLIGER',
       'FINANZ_HAUSBAUER', 'HEALTH_TYP', 'RETOURTYP_BK_S',
       'SEMIO_SOZ', 'SEMIO_FAM', 'SEMIO_REL', 'SEMIO_MAT', 'SEMIO_VERT',
       'SEMIO_LUST', 'SEMIO_ERL', 'SEMIO_KULT', 'SEMIO_RAT', 'SEMIO_KRIT',
       'SEMIO_DOM', 'SEMIO_KAEM', 'SEMIO_PFLICHT', 'SEMIO_TRADV',
       'ANZ_PERSONEN', 'ANZ_TITEL', 'HH_EINKOMMEN_SCORE', 'W_KEIT_KIND_HH',
       'WOHNDAUER_2008', 'ANZ_HAUSHALTE_AKTIV', 'ANZ_HH_TITEL',
       'KONSUMNAEHE', 'MIN_GEBAEUDEJAHR', 'KBA05_ANTG1', 'KBA05_ANTG2',
       'KBA05_ANTG3', 'KBA05_ANTG4', 'KBA05_GBZ', 'BALLRAUM', 'EWDICHTE',
       'INNENSTADT', 'GEBAEUDETYP_RASTER', 'KKK', 'MOBI_REGIO',
       'ONLINE_AFFINITAET', 'REGIOTYP', 'KBA13_ANZAHL_PKW', 'PLZ8_ANTG1',
       'PLZ8_ANTG2', 'PLZ8_ANTG3', 'PLZ8_ANTG4', 'PLZ8_HHZ', 'PLZ8_GBZ',
       'ARBEIT', 'ORTSGR_KLS9', 'RELAT_AB']])

#get useful categorical columns
print('make separate df of categorical col only')

azdias_no_miss_cat_keep=pd.DataFrame(azdias_no_miss[['ANREDE_KZ', 'CJT_GESAMTTYP', 'FINANZTYP',
       'GFK_URLAUBERTYP', 'GREEN_AVANTGARDE',
       'LP_FAMILIE_GROB', 'LP_STATUS_GROB',
       'NATIONALITAET_KZ', 'SHOPPER_TYP', 'SOHO_KZ', 
       'VERS_TYP', 'ZABEOTYP', 'GEBAEUDETYP',
       'OST_WEST_KZ', 'CAMEO_DEUG_2015']])

#letter to numeric category
print('translate W/O')
azdias_no_miss_cat_keep.loc[azdias_no_miss_cat_keep['OST_WEST_KZ']=='O','OST_WEST_KZ']=0
azdias_no_miss_cat_keep.loc[azdias_no_miss_cat_keep['OST_WEST_KZ']=='W','OST_WEST_KZ']=1

categorical_col_keep=['ANREDE_KZ', 'CJT_GESAMTTYP', 'FINANZTYP',
       'GFK_URLAUBERTYP', 'GREEN_AVANTGARDE',
       'LP_FAMILIE_GROB', 'LP_STATUS_GROB',
       'NATIONALITAET_KZ', 'SHOPPER_TYP', 'SOHO_KZ', 
       'VERS_TYP', 'ZABEOTYP', 'GEBAEUDETYP',
       'OST_WEST_KZ', 'CAMEO_DEUG_2015']

#create the column name for encoded data
print('make one hot col list')
column_name_cat=[]
for col_ in categorical_col_keep:
    try:
        list_value_sort=sorted(list(azdias_no_miss_cat_keep[col_].value_counts().index))
        len_=len(list_value_sort)
        #print(col_,list(azdias_no_miss_cat_keep[col_].value_counts().index))
        print(col_,list_value_sort,len_)
        for i in range(len_):
            column_name_cat.append(str(col_)+'_'+str(int(list_value_sort[i])))
            
            
    except:
        pass
print('make one hot coding')
encode__ = OneHotEncoder()

azdias_no_miss_cat_keep_encode=encode__.fit_transform(azdias_no_miss_cat_keep)

print('make separate df of categorical after one hot code')
azdias_no_miss_cat_keep_encode_df=pd.DataFrame(azdias_no_miss_cat_keep_encode.toarray(),columns=column_name_cat)

#get useful mix columns

print('make separate df of mix')
azdias_no_miss_mix_keep=pd.DataFrame(azdias_no_miss[['PRAEGENDE_JUGENDJAHRE', 'CAMEO_INTL_2015']])

#translation dictionary

dic_decade={1.0:0.0, 2.0:0.0, 3.0:1.0, 4.0:1.0, 5.0:2.0, 6.0:2.0, 7.0:2.0, 8.0:3.0, 9.0:3.0, 10.0:4.0, 11.0:4.0, 12.0:4.0, 13.0:4.0, 14.0:5.0, 15.0:5.0}
dic_movement={1.0:0.0, 2.0:1.0, 3.0:0.0, 4.0:1.0, 5.0:0.0, 6.0:1.0, 7.0:1.0, 8.0:0.0, 9.0:1.0, 10.0:0.0, 11.0:1.0, 12.0:0.0, 13.0:1.0, 14.0:0.0, 15.0:1.0}

temp_=azdias_no_miss_mix_keep['PRAEGENDE_JUGENDJAHRE']

#encoding mix data
print('make encoding mix')
for k_,v_ in dic_decade.items():
    azdias_no_miss_mix_keep.loc[round(temp_,1)==round(k_,1),'youth_decade']=round(v_,1)
    
for k_,v_ in dic_movement.items():
    azdias_no_miss_mix_keep.loc[round(temp_,1)==round(k_,1),'youth_movement']=round(v_,1)


azdias_no_miss_mix_keep['CAMEO_wealth']=azdias_no_miss_mix_keep['CAMEO_INTL_2015'].apply(lambda x: round(x/10,0))
azdias_no_miss_mix_keep['CAMEO_family']=azdias_no_miss_mix_keep['CAMEO_INTL_2015'].apply(lambda x: float(x%10))

#only keep encoded data
print('make encoding mix, drop unnecessary stuff after encode')
azdias_no_miss_mix_keep_encode=azdias_no_miss_mix_keep.drop(['PRAEGENDE_JUGENDJAHRE','CAMEO_INTL_2015'],axis=1)

#attach all encoded (mix, categorical), and numeric data into one data frame
print('make all three df into one after encoding')
azdias_no_miss_encoded_new=pd.concat([azdias_no_miss_num_keep,azdias_no_miss_cat_keep_encode_df,azdias_no_miss_mix_keep_encode], axis=1)

#drop any null if any still exist
print('last row cleanup in case')
azdias_no_miss_encoded_new=azdias_no_miss_encoded_new.dropna()
print('write final to file')

azdias_no_miss_encoded_new.to_csv('azdias_no_miss_encoded.csv',index=False)  


