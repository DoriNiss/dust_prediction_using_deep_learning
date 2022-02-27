def get_pm10_pm25_translated_station_name_from_hebrew(hebrew_name):
    tranlation_dict = get_tranlation_dict_pm10_pm25()
    if hebrew_name not in tranlation_dict.keys():
        return None
    return tranlation_dict[hebrew_name]

def get_tranlation_dict_pm10_pm25():
    translation_dict = {
        "עפולה, עפולה": "AFULA",
        "תל אביב-יפו, רחוב לחי": "AMIEL",
        "אריאל, אריאל": "ARIEL",
        "ירושלים, רחוב בר אילן": "BAR_ILAN", 
        "באר שבע, שכונה ו": "BEER_SHEVA",
        "בית שמש, בית שמש": "BEIT_SHEMESH", 
        "כרמיאל, גליל מערבי": "WEST_GALIL",
        "ירושלים, בקעה": "EFRATA", 
        "אלון שבות, גוש עציון": "GUSH_EZION",
        "חולון, חולון": "HOLON", 
        "תל אביב-יפו, רחוב יהודה המכבי": "IRONID", 
        "רחובות, רחובות": "REHOVOT", 
        "בני ברק, רחוב ז'בוטינסקי": "REMEZ", 
        "בני ברק, כביש 4": "KVISH4", 
        "ירושלים, כיכר ספרא": "SAFRA", 
        "תל אביב-יפו, אוניברסיטה": "YAD_AVNER", 
        "פתח תקווה, רחוב גיסין": "EHAD_HAAM", 
        "רמת גן, יד לבנים": "YAD_LEBANIM", 
        "תל אביב-יפו, הצפון החדש": "ANTOKOLSKY", 
        "תל אביב-יפו, שיכון ל": "SHIKUN_L", 
        "ניר גלים, ניר גלים 1": "NIR_GALIM", 
        "מודיעין, מודיעין": "MODEIN_IEC", 
        "יבנה, יבנה": "YAVNE_CITY", 
        "חיפה, פארק הכרמל": "PARK_HACARMEL", 
        "בת הדר, בת הדר": "BAT_HADAR", 
        "תל אביב-יפו, לב תל אביב": "PETAH_TIKVA_ROAD", 
        "חדרה, תחנת הכוח אורות רבין": "HADERA_B", #proximal location 
        "חיפה, נווה שאנן": "NAVE_SHANAAN", 
        "נשר, נשר": "NESHER", 
        "קריית אתא, מרכז העיר": "KIRYAT_ATA", 
        "חיפה, צ'ק פוסט": "IGUD", 
        "קריית אתא, קריית בנימין": "KIRYAT_BINYAMIN", 
        "פרדס חנה, פרדס חנה": "PARDES_HANA", 
        "אום אל פאחם, כביש 65": "UM_EL_FAHEM_C", 
        "אשדוד, אזור תעשייה קלה": "ASHDOD_IGUD", 
        "גדרה, גדרה": "GEDERA", 
        "יד בנימין, יד בנימין": "YAHALOM", #proximal location 
        "אשקלון, גן הורדים": "ASHKELON", 
        "קריית מלאכי, קריית מלאכי": "KIRYAT_MALAHI", 
        "שדרות, שדרות": "SDEROT", 
        "גברעם, גברעם": "GVARAAM", 
        "שדה יואב, שדה יואב": "SDE_YOAV", 
        "ניר ישראל, ניר ישראל": "NIR_ISRAEL", 
        "כרמיה, כרמיה": "CARMIYA", 
        "ארז, ארז": "EREZ", 
        "חיפה, אחוזה": "AHUZA_G",
        "קיריית טבעון, קריית טבעון": "KIRYAT_TIVON", 
        "ראשון לציון, רחוב הרצל": "RISHON_LEZION", 
        "מועצה אזורית חוף הכרמל, טורבינת הגז קיסריה": "CEASARIA", 
        "רעננה, רחוב אחוזה": "RAANANA", 
        "ערד, נגב מזרחי": "EAST_NEGEV", 
        "אשקלון, שמשון": "ASHKELON_SOUTH", 
        "תל אביב-יפו, רחוב יפת": "YEFET_YAFO", 
        "סדום, לוט": "LOT", 
        "סדום, נחל אשלים": "ASHALIM", 
        "נאות הכיכר, נאות הכיכר": "NEOT_HAKIKAR", 
        "חיפה, עצמאות חיפה": "ATZMAUT_B", 
        "ירושלים, רוממה": "TAHANA_MERKAZIT_JERUSALEM", 
        "אילת, שכונת שחמון": "EILAT6", 
        "מבקיעים, מבקיעים": "MAVKIIM", 
        "אום אל קוטוף, אום אל קוטוף": "UM_EL_KUTUF", 
        "כפר מסריק, כפר מסריק החדשה": "KFAR_MASARIK", 
        "כפר מנחם, שכונת ההרחבה": "DALYA", 
        'יד רמב"ם, יד רמב"ם החדשה': "YAD_RAMBAM_2", 
        "מועצה אזורית חוף הכרמל, כפר הנוער שפיה": "SHFEYA", 
        "תל אביב-יפו, תחנת רכבת השלום": "RAKEVET_HASHALOM", 
        "בני עטרות, בני עטרות": "BNEI_ATAROT", 
        "בני דרום, בני דרום": "BNEI_DAROM", 
        "בית חשמונאי, בית חשמונאי": "BEIT_HASHMONAY", 
        "אור יהודה, אור יהודה": "BEN_GURION_AIRPORT", #proximal location 
        "כפר סבא, כפר סבא": "KFAR_SABA", 
        "קציר, קציר": "KAZIR_NEW", 
        "קריית ביאליק, דרום": "KIRYAT_BIALIK", 
        "קריית מוצקין, נווה גנים": "BEGIN", 
        'אשדוד, רובע ט"ו החדשה': "OFEK", 
        "חולון, תחנת רכבת קוממיות": "BEIT_RIVKA", #proximal location 
        "חיפה, הרצל - בלפור": "HADAR", 
        "חדרה, חפציבה": "HEFZIBA", 
        "תל אביב-יפו, רחוב לוינסקי": "TAHANA_MERKAZIT", 
        "ברטעה, ברטעה": "BARTAA", 
        "ברקאי, ברקאי החדשה": "BARKAI", 
        "חיפה, קרית חיים - מערבית": "KIRYAT_HAIM", 
    }
    return translation_dict


"""
~~~~~~~~~~~~~~~~~~~~~
STATIONS' TRANSLATION
~~~~~~~~~~~~~~~~~~~~~



Duplicates / not existing / comments:
x: 205492, y: 748547, PM10  KIRYAT_HAIM_REGAVIM (KIRYAT_HAIM)
x: 207828, y: 748341, PM2.5 AKO (KIRYAT_HAIM)
x: 198159, y: 747329, PM10  CARMEL_ZARFATI שפרינצק
x: 207100, y: 746464, PM2.5 KAKAL קרית ביאליק / קרית אתא
x: 192141, y: 706027, PM10  HADERA_B - proximated to חדרה, תחנת הכוח אורות רבין
x: 164954, y: 631264, PM2.5 ROVA_TV (OFEK)
x: 166336, y: 635778, PM2.5 YAHALOM proximated to יד בנימין, יד בנימין
x: 190355, y: 655382, PM2.5 BEN_GURION_AIRPORT proximated to אור יהודה, אור יהודה
x: 187360, y: 655670, PM2.5 BEIT_RIVKA proximated to חולון, תחנת רכבת קוממיות

x: 210861, y: 746333, PM2.5 KIRYAT_ATA
x: 200582, y: 746022, PM10  SHUK
x: 197620, y: 744760, PM10  CARMELIA
x: 209342, y: 744564, PM2.5 KIRYAT_BINYAMIN
x: 201240, y: 743960, PM10  IZRAELIA
x: 202450, y: 743960, PM10  NAVE_YOSEF
x: 200530, y: 743920, PM10  ROMMEMA
x: 204108, y: 743853, PM2.5 IGUD
x: 202285, y: 743626, PM2.5 NAVE_SHANAAN
x: 204317, y: 741718, PM2.5 NESHER
x: 203682, y: 738118, PM10  PARK_HACARMEL
x: 197768, y: 721586, PM2.5 SHFEYA
x: 203752, y: 698998, PM2.5 MAGAL
x: 188084, y: 667486, PM2.5 EHAD_HAAM
x: 182301, y: 664067, PM10  GIVATAIM
x: 194757, y: 648078, PM10  GIMZO
x: 201350, y: 646186, PM10  MODEIN
x: 192484, y: 639345, PM10  KARMEY_YOSSEF
x: 166858, y: 636000, PM2.5 ORT
x: 218947, y: 634660, PM10  KVISH9
x: 208844, y: 634288, PM2.5 NAVE_ILAN
x: 220325, y: 632370, PM2.5 KARON_KIACH
x: 220183, y: 632369, PM2.5 AGRIPAS
x: 220763, y: 629391, PM2.5 EFRATA
x: 184410, y: 617120, PM10  HAKFAR_HAYAROK
x: 194714, y: 385170, PM10  EILAT_IEC

x: NaN, y: NaN, PM10  EVEN_AMI
x: NaN, y: NaN, PM2.5 AZUR
x: NaN, y: NaN, PM2.5 HATIKVA

Missing PM10 stations:
נתניה, קריית השרון PM10
ירושלים, אזור תעשייה עטרות PM10  (unknown location)
אשלים, אשלים PM10
מצפה נטופה, מצפה נטופה PM10
טורעאן, טורעאן החדשה PM10
אלעד, אלעד PM10
אלעד, רובע א PM10
רמלה, שכונת האמנים PM10
כסייפה, כסייפה PM10


"""