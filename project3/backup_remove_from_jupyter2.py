def customer_clean_encode(customer,feature__customer):
    
    customer=customer
    feature_=feature__customer

    feature__n = copy.deepcopy(feature_)
    feature__n=feature__n.drop(['information_level','type'],axis=1) #created new data frame to be able to pla with cleaning
    feature__n['missing_or_unknown_']=feature__n['missing_or_unknown']
    # get the splitted encode
    for s in range(np.shape(feature__n)[0]):
        #print(len(str(feature__n['missing_or_unknown'][s])))
        if(len(str(feature__n['missing_or_unknown'][s]))==2):   #no clean out needed is 2 character []
            feature__n['missing_or_unknown_'][s]='pass'
            #print("['pass']")
        else:

            snew=str(feature__n['missing_or_unknown'][s]).replace('[','').replace(']','').split(',')    #get the values in real list
            #print(snew)
            feature__n['missing_or_unknown_'][s]=snew

    feature__n=feature__n.drop(['missing_or_unknown'],axis=1) #drop extra col

    customer_ = copy.deepcopy(customer)

    print('start converting to nan')
    for i,column_ in enumerate(customer_.columns):
        if(i<86):
            missing_attr=feature__n[feature__n['attribute']==column_]['missing_or_unknown_'].values[0]
            if(missing_attr!='pass'):
                for j in range(len(missing_attr)):
                    try:
                        customer_.loc[customer_[column_]==int(missing_attr[j]),column_]=np.nan
                    except:
                        print(column_)
                        customer_.loc[customer_[column_]=='X',column_]=np.nan
                        customer_.loc[customer_[column_]=='XX',column_]=np.nan

    print('finish converting to nan')
    #drop the columns with outlier data that was evident from original data
    customer_out_remove = copy.deepcopy(customer_)
    print('drop outliers col')
    customer_out_remove=customer_out_remove.drop(['AGER_TYP','GEBURTSJAHR','TITEL_KZ','ALTER_HH','KK_KUNDENTYP','KBA05_BAUMAX'],axis=1)

    print('drop outliers row')
    customer_no_miss=customer_out_remove.dropna()

    print('write no miss data to file')
    customer_no_miss.to_csv('customer_no_miss_data.csv',index=False) 

    #get useful numeric columns
    print('make separate df of numeric col only')
    customer_no_miss_num_keep=pd.DataFrame(customer_no_miss[['ALTERSKATEGORIE_GROB', 'FINANZ_MINIMALIST', 'FINANZ_SPARER',
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

    customer_no_miss_cat_keep=pd.DataFrame(customer_no_miss[['ANREDE_KZ', 'CJT_GESAMTTYP', 'FINANZTYP',
           'GFK_URLAUBERTYP', 'GREEN_AVANTGARDE',
           'LP_FAMILIE_GROB', 'LP_STATUS_GROB',
           'NATIONALITAET_KZ', 'SHOPPER_TYP', 'SOHO_KZ', 
           'VERS_TYP', 'ZABEOTYP', 'GEBAEUDETYP',
           'OST_WEST_KZ', 'CAMEO_DEUG_2015']])

    #letter to numeric category
    print('translate W/O')
    customer_no_miss_cat_keep.loc[customer_no_miss_cat_keep['OST_WEST_KZ']=='O','OST_WEST_KZ']=0
    customer_no_miss_cat_keep.loc[customer_no_miss_cat_keep['OST_WEST_KZ']=='W','OST_WEST_KZ']=1

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
            list_value_sort=sorted(list(customer_no_miss_cat_keep[col_].value_counts().index))
            len_=len(list_value_sort)
            #print(col_,list(customer_no_miss_cat_keep[col_].value_counts().index))
            print(col_,list_value_sort,len_)
            for i in range(len_):
                column_name_cat.append(str(col_)+'_'+str(int(list_value_sort[i])))


        except:
            pass
    print('make one hot coding')
    encode__ = OneHotEncoder()

    customer_no_miss_cat_keep_encode=encode__.fit_transform(customer_no_miss_cat_keep)

    print('make separate df of categorical after one hot code')
    customer_no_miss_cat_keep_encode_df=pd.DataFrame(customer_no_miss_cat_keep_encode.toarray(),columns=column_name_cat)

    #get useful mix columns

    print('make separate df of mix')
    customer_no_miss_mix_keep=pd.DataFrame(customer_no_miss[['PRAEGENDE_JUGENDJAHRE', 'CAMEO_INTL_2015']])

    #translation dictionary

    dic_decade={1.0:0.0, 2.0:0.0, 3.0:1.0, 4.0:1.0, 5.0:2.0, 6.0:2.0, 7.0:2.0, 8.0:3.0, 9.0:3.0, 10.0:4.0, 11.0:4.0, 12.0:4.0, 13.0:4.0, 14.0:5.0, 15.0:5.0}
    dic_movement={1.0:0.0, 2.0:1.0, 3.0:0.0, 4.0:1.0, 5.0:0.0, 6.0:1.0, 7.0:1.0, 8.0:0.0, 9.0:1.0, 10.0:0.0, 11.0:1.0, 12.0:0.0, 13.0:1.0, 14.0:0.0, 15.0:1.0}

    temp_=customer_no_miss_mix_keep['PRAEGENDE_JUGENDJAHRE']

    #encoding mix data
    print('make encoding mix')
    for k_,v_ in dic_decade.items():
        customer_no_miss_mix_keep.loc[round(temp_,1)==round(k_,1),'youth_decade']=round(v_,1)

    for k_,v_ in dic_movement.items():
        customer_no_miss_mix_keep.loc[round(temp_,1)==round(k_,1),'youth_movement']=round(v_,1)


    customer_no_miss_mix_keep['CAMEO_wealth']=customer_no_miss_mix_keep['CAMEO_INTL_2015'].apply(lambda x: round(x/10,0))
    customer_no_miss_mix_keep['CAMEO_family']=customer_no_miss_mix_keep['CAMEO_INTL_2015'].apply(lambda x: float(x%10))

    #only keep encoded data
    print('make encoding mix, drop unnecessary stuff after encode')
    customer_no_miss_mix_keep_encode=customer_no_miss_mix_keep.drop(['PRAEGENDE_JUGENDJAHRE','CAMEO_INTL_2015'],axis=1)

    #attach all encoded (mix, categorical), and numeric data into one data frame
    print('make all three df into one after encoding')
    customer_no_miss_encoded_new=pd.concat([customer_no_miss_num_keep,customer_no_miss_cat_keep_encode_df,customer_no_miss_mix_keep_encode], axis=1)

    #drop any null if any still exist
    print('last row cleanup in case')
    customer_no_miss_encoded_new=customer_no_miss_encoded_new.dropna()
    print('write final to file')

    customer_no_miss_encoded_new.to_csv('customer_no_miss_encoded.csv',index=False)  
    
    return customer_no_miss_encoded_new


