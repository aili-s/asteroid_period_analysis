import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, fft
from astropy.timeseries import LombScargle
import warnings
warnings.filterwarnings('ignore')

#============================================================
#============================================================
#БЛОК 1: ПІДГОТОВКА ДАНИХ
#============================================================
#============================================================

print("ВИБІР ОДНІЄЇ ОПОЗИЦІЇ")

#Відомі періоди: якщо мало даних, то період не точний, чим більше даних тим точніший період
#df = pd.read_csv(r"D:\Робочий стіл\диплом\тестові об'єкти\table_1-Ceres20000001.csv")#+
#df = pd.read_csv(r"D:\Робочий стіл\диплом\тестові об'єкти\table_4-Vesta20000004.csv")#+
#df = pd.read_csv(r"D:\Робочий стіл\диплом\тестові об'єкти\table_15-Eunomia20000015.csv")#+- завелика похибка 

# EUNOMIA ASTEROID FAMILY
#left side
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left\1-10\table_104059-2000-EZ1720104059.csv") #15.3% S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left\1-10\table_106903-2000-YN4520106903.csv") #--
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left\1-10\table_107643-2001-EY1720107643.csv") #46.2
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left\1-10\table_107758-2001-FM3720107758.csv") #23.6
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left\1-10\table_110795-2001-UN3720110795.csv") #22.1
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left\1-10\table_111721-2002-CN4220111721.csv") #25.3
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left\1-10\table_115204-2003-SV11620115204.csv") #79.3
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left\1-10\table_118366-1999-GK20118366.csv") #12.9 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left\1-10\table_119443-2001-TF14720119443.csv") #12.1 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left\1-10\table_122252-2000-OM3820122252.csv") #15.9 
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left\table_122695-2000-SY1320122695.csv") #63.2
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left\table_125498-2001-WU3020125498.csv") #21.0
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left\table_126467-2002-CZ3920126467.csv") #40.0
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left\table_127869-2003-FW12120127869.csv") #26.2
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left\table_132583-2002-JH11920132583.csv") #10.3 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left\table_135675-2002-NU3120135675.csv") #22.0
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left\table_136626-1994-PX1820136626.csv") #11.9 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left\table_137055-1998-VN3620137055.csv") #4.9 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left\table_142627-2002-TZ16920142627.csv") #13.3 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left\table_144280-2004-CH10120144280.csv") #--

#S-type Carvano
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\table_79565-1998-QV7520079565.csv") #5.6 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\table_87121-2000-LG3520087121.csv") #34.4
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\table_89458-2001-XV520089458.csv") #26.6
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\table_91086-1998-FF12020091086.csv") #21.2
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\table_104059-2000-EZ1720104059.csv") #15.3 S ()
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\table_118366-1999-GK20118366.csv") #12.9 S ()
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\table_119443-2001-TF14720119443.csv") #12.1 S ()
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\table_122695-2000-SY1320122695.csv") #63.2 ()
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\table_142627-2002-TZ16920142627.csv") #13.3 S ()
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\table_149433-2003-BL6620149433.csv") #14.2  S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\table_153937-2001-YG9720153937.csv") #40.7 
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\table_157779-7582-P-L20157779.csv") #11.0 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\table_179126-2001-SC27120179126.csv") #36.5
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\table_187980-2001-QP28820187980.csv") #9.5 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\table_187997-2001-SC16120187997.csv") #11.5 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\table_188479-2004-NN2420188479.csv") #27.0
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\table_189538-2000-QK11820189538.csv") #1.5 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\table_203164-2000-XY1520203164.csv") #42.6
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\table_260746-2005-MU620260746.csv") #12.1 S

#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\add\table_119223-2001-QB22020119223.csv") #14.0 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\add\table_393285-2013-YE620393285.csv") #8.7 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\add\table_257576-1999-CN8220257576.csv") #7.6 V
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\add\table_351399-2005-ET24020351399.csv") #19.3
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\add\1-10\table_351399-2005-ET24020351399.csv") #19.3
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\add\1-10\table_225836-2001-XE8720225836.csv") #8.1 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\add\1-10\table_370161-2001-YB13320370161.csv") #14.0 C
df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\add\1-10\table_225938-2002-AV18420225938.csv") #12.4 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\add\1-10\table_286917-2002-PV10020286917.csv") #13.1 C
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\add\1-10\table_192459-1998-DS620192459.csv") #34.6
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\add\1-10\table_284223-2006-DE1020284223.csv") #10.3 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\add\1-10\table_353026-2009-BJ18820353026.csv") #19.7
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\add\1-10\table_254591-2005-GJ6920254591.csv") #9.8 V
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\add\1-10\table_164818-1999-RR4120164818.csv") #14.9 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\add\11-16\table_380838-2006-AD5620380838.csv")
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\add\11-16\table_371991-2008-GU12920371991.csv") #19.8
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\add\11-16\table_190560-2000-SU8620190560.csv") #58.3
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\add\11-16\table_339109-2004-RT21620339109.csv") #42.0
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\add\11-16\table_241989-2002-LT5920241989.csv") #17.3
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\add\11-16\table_273637-2007-DN5120273637.csv") #16.3
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\add\11-16\table_119768-2001-YA13620119768.csv") #14.9 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\add\11-16\table_149303-2002-TT29520149303.csv") #12.1 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\add\11-16\table_143465-2003-CT20143465.csv") #28.1
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Left 1\add\11-16\table_75766-2000-AG18520075766.csv") #16.7

#right side
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Right\1-10\table_107701-2001-FS1720107701.csv") #14.7 LS
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Right\1-10\table_110869-2001-UA9420110869.csv") #15.2 LS
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Right\1-10\table_116015-2003-WE8320116015.csv") #3.9 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Right\1-10\table_117206-2004-RC19320117206.csv") #30.9
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Right\1-10\table_143112-2002-XW2420143112.csv") #21.7
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Right\1-10\table_146643-2001-UZ7420146643.csv") #18.0
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Right\1-10\table_149470-2003-EU1220149470.csv") #23.0
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Right\1-10\table_150504-2000-QT19020150504.csv") #10.8 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Right\1-10\table_153847-2001-XU4220153847.csv") #20.4
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Right\1-10\table_156526-2002-CT25420156526.csv") #7.8 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Right\table_158510-2002-EW10320158510.csv") #72.7
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Right\table_161049-2002-JV1220161049.csv") #23.4
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Right\table_161064-2002-JJ10720161064.csv") #18.5
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Right\table_162090-1998-PP20162090.csv") #33.5
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Right\table_176917-2002-VP8820176917.csv") #34.6
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Right\table_182618-2001-UT9120182618.csv") #--
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Right\table_185889-2000-QK19220185889.csv") #7.2 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Right\table_185899-2000-RC8020185899.csv") #12.7 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Right\table_189044-2000-PB320189044.csv") --
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Right\table_190115-2004-VY1320190115.csv") #5.9 S

#S-type Carvano
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\table_199627-2006-FV5220199627.csv") #30.6
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\table_211702-2003-WK16920211702.csv") #29.2
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\table_348319-2005-CH220348319.csv") #15.0 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\table_354051-2001-SP29620354051.csv") #3.2 S

#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\table_136265-2003-YN7120136265.csv") #--
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\table_78297-2002-PH5420078297.csv") #15.4 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\table_89873-2002-CT15820089873.csv") #28.6
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\table_258480-2002-AY2220258480.csv") #37.6
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\table_206687-2003-YC14820206687.csv") #14.0 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\table_206576-2003-VE820206576.csv") #12.1 C
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\table_212273-2005-JR14720212273.csv") #2.6 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\table_189064-2000-WL1420189064.csv") #4.6 LS
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\table_170662-2003-YD14120170662.csv") #5.1 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\table_137118-1999-BV420137118.csv") #27.2
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\1-10\table_207551-2006-LS320207551.csv") #17.9
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\1-10\table_282008-2011-HV5720282008.csv") #26.2
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\1-10\table_259325-2003-FB7220259325.csv") #16.9
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\1-10\table_249897-2001-SD1120249897.csv") #7.4 C
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\1-10\table_225998-2002-DA820225998.csv") #59.2
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\1-10\table_244773-2003-SL14420244773.csv") #39.0
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\1-10\table_222325-2000-UH1520222325.csv") #23.9
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\1-10\table_218670-2005-SC22120218670.csv") #15.1 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\1-10\table_237100-2008-TE7320237100.csv") #10.6 C
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\1-10\table_217999-2001-XH5420217999.csv") #8.5 C
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\11-20\table_256174-2006-VS6720256174.csv") #19.3
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\11-20\table_240647-2005-BU2620240647.csv") #14.7 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\11-20\table_239765-2010-CO3420239765.csv") #6.5 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\11-20\table_238237-2003-UV22020238237.csv") #13.9 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\11-20\table_329005-2010-XO5720329005.csv") #12.8 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\11-20\table_204084-2003-WG3920204084.csv") #12.0 S
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\11-20\table_287071-2002-RZ2820287071.csv") #48.4
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\11-20\table_285854-2001-FH19020285854.csv") #15.1 V
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\11-20\table_281252-2007-NX520281252.csv") #21.4
#df = pd.read_csv(r"D:\Робочий стіл\диплом\Eunomnia\Rigth 1\add\11-20\table_270579-2002-JM9920270579.csv") #58.6

df['obsdate'] = pd.to_datetime(df['obsdate'])

df['year'] = df['obsdate'].dt.year
df['month'] = df['obsdate'].dt.month

print("Розподіл спостережень за роками:")
year_counts = df['year'].value_counts().sort_index()
print(year_counts)

# Знаходимо рік з найбільшою кількістю спостережень
best_year = year_counts.idxmax()
year_data = df[df['year'] == best_year].copy()

print(f"\nНайкращий рік для аналізу: {best_year} ({len(year_data)} точок)")

print(f"\nРозподіл за місяцями у {best_year}:")
month_counts = year_data['month'].value_counts().sort_index()
print(month_counts)

# Знаходимо діапазон місяців з найбільшою кількістю спостережень
if len(month_counts) >= 3:
    top_months = month_counts.nlargest(3).index.sort_values()
    synodic_data = year_data[year_data['month'].isin(top_months)].copy()
    
    print(f"\nСинодичний період: місяці {list(top_months)}")
    print(f"Кількість точок у синодичному періоді: {len(synodic_data)}")
    
elif len(month_counts) == 2:
    top_months = month_counts.nlargest(2).index.sort_values()
    synodic_data = year_data[year_data['month'].isin(top_months)].copy()
    
    print(f"\nСинодичний період: місяці {list(top_months)}")
    print(f"Кількість точок у синодичному періоді: {len(synodic_data)}")
    
else:
    print("\nДані тільки за один місяць, аналізуємо розподіл за днями...")
    
    daily_counts = year_data['obsdate'].dt.date.value_counts().sort_index()
    
    if len(daily_counts) >= 10:
        dates_sorted = daily_counts.index.sort_values()
        date_to_index = {date: i for i, date in enumerate(dates_sorted)}
        
        best_range = None
        max_points = 0
        
        # Перебираємо всі можливі діапазони довжиною 30-90 днів
        for start_date in dates_sorted:
            start_idx = date_to_index[start_date]
            
            for days in range(30, 91, 5):
                end_date = start_date + pd.Timedelta(days=days)
                if end_date > dates_sorted[-1]:
                    continue
                    
                if end_date in date_to_index:
                    end_idx = date_to_index[end_date]
                    date_range = dates_sorted[start_idx:end_idx+1]
                    total_points = daily_counts[date_range].sum()
                    
                    if total_points > max_points:
                        max_points = total_points
                        best_range = (start_date, end_date)
        
        if best_range:
            start_date, end_date = best_range
            synodic_data = year_data[
                (year_data['obsdate'].dt.date >= start_date) & 
                (year_data['obsdate'].dt.date <= end_date)
            ].copy()
            
            print(f"Синодичний період: {start_date} до {end_date} ({len(synodic_data)} точок)")
            print(f"Тривалість: {(end_date - start_date).days} днів")
        else:
            # Якщо не знайшли хорошого діапазону, беремо весь рік
            synodic_data = year_data.copy()
            print(f"Використовуємо весь {best_year} рік ({len(synodic_data)} точок)")
    else:
        # Якщо замало днів, беремо весь рік
        synodic_data = year_data.copy()
        print(f"Замало днів з даними, використовуємо весь {best_year} рік ({len(synodic_data)} точок)")

# Додатково фільтр за фазовим кутом (+ видалення екстремальних значень)
phase_q25 = synodic_data['phase'].quantile(0.25)
phase_q75 = synodic_data['phase'].quantile(0.75)
iqr = phase_q75 - phase_q25
phase_lower = phase_q25 - 1.5 * iqr
phase_upper = phase_q75 + 1.5 * iqr

synodic_data = synodic_data[
    (synodic_data['phase'] >= phase_lower) & 
    (synodic_data['phase'] <= phase_upper)
].copy()

print(f"\nЗа фазовим кутом: {len(synodic_data)} точок")
print(f"Фазовий кут: {synodic_data['phase'].min():.1f}° - {synodic_data['phase'].max():.1f}°")

# Візуалізація вибраного синодичного періоду
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
# Розподіл за місяцями
month_data = synodic_data.groupby('month').size()
plt.bar(month_data.index, month_data.values, alpha=0.7, color='blue')
plt.xlabel('Місяць')
plt.ylabel('Кількість точок')
plt.title(f'Розподіл точок за місяцями ({best_year})')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Фазовий кут vs час
plt.scatter(synodic_data['obsjd'] - synodic_data['obsjd'].min(), 
           synodic_data['phase'], alpha=0.5, s=20)
plt.xlabel('Час (дні від початку)')
plt.ylabel('Фазовий кут (°)')
plt.title('Фазовий кут протягом синодичного періоду')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# =============================================================================
# ПРИВЕДЕННЯ ДО ОДИНИЧНОЇ ВІДСТАНІ З УРАХУВАННЯМ ПОХИБОК
# =============================================================================
# =============================================================================

print("ПРИВЕДЕННЯ ДО ОДИНИЧНОЇ ВІДСТАНІ")

# Словник похибок для кожного фільтра. Похибку розраховано центром Цвіккі
error_dict = {'zg': 0.003, 'zr': 0.002, 'zi': 0.01}

def reduce_to_unit_distance_with_errors(vmag, r, delta, phase_angle, filter_code, G=0.035):

    # Базова корекція відстані та фазового кута
    H_distance = vmag - 5 * np.log10(r * delta)
    beta = G  # mag/degree
    H_corrected = H_distance - beta * phase_angle
    
    # Похибка виміру
    measurement_error = error_dict.get(filter_code, 0.01)
    
    return H_corrected, measurement_error

H_values = []
H_errors = []

for idx, row in synodic_data.iterrows():
    H, err = reduce_to_unit_distance_with_errors(
        row['vmag'], row['sun_dist'], row['geo_dist'], 
        row['phase'], row['filtercode'], G=0.035
    )
    H_values.append(H)
    H_errors.append(err)

synodic_data['H_reduced'] = H_values
synodic_data['H_error'] = H_errors

print("Приведена зоряна величина:")
print(f"Середнє H: {synodic_data['H_reduced'].mean():.3f} ± {synodic_data['H_reduced'].std():.3f}")

# =============================================================================
# =============================================================================
# ВИБІР ДАНИХ ДЛЯ АНАЛІЗУ ПЕРІОДУ
# =============================================================================
# =============================================================================

print("ПІДГОТОВКА ДАНИХ ДЛЯ АНАЛІЗУ ПЕРІОДУ")

# Вибираємо фільтр з найбільшою кількісттю даних 
period_data = synodic_data[synodic_data['filtercode'] == 'zr'].copy()
if len(period_data) < 10:
    period_data = synodic_data[synodic_data['filtercode'] == 'zr'].copy()

print(f"Фільтр {period_data['filtercode'].iloc[0]} ({len(period_data)} точок)")

# Сортування за часом
period_data = period_data.sort_values('obsjd')
time = period_data['obsjd'].values - 0.005776 * row['geo_dist']
magnitude = period_data['H_reduced'].values
errors = period_data['H_error'].values
time_norm = time - time.min()


print(f"Часовий діапазон: {time_norm.max()*24:.2f} годин")

# =============================================================================
# =============================================================================
# ФУНКЦІЇ
# =============================================================================
# =============================================================================

def advanced_period_search(time, magnitude, errors=None):

    print("Застосовуємо розширений пошук періоду...")
    
    # Діапазон періодів для пошуку (2-12 годин, спричинено бар'єрами та малим розміром. Окрім, запобігаю створенню хибних гармонік)
    period_range_hours = (2.2, 12.0)
    
    # Метод 1: Ломб-Скардж
    frequencies = np.linspace(1/period_range_hours[1], 1/period_range_hours[0], 10000)
    
    try:
        power = LombScargle(time, magnitude).power(frequencies)
        period_ls = 1/frequencies[np.argmax(power)]
        print(f"Ломб-Скардж: {period_ls:.4f} годин")
    except:
        period_ls = 0.0
        print("Ломб-Скардж не вдався")
    
    # Метод 2: FFT аналіз
    try:
        # Інтерполяція. 312 коефіцієнтів
        n_points = min(312, len(time))
        time_uni = np.linspace(time.min(), time.max(), n_points)
        mag_uni = np.interp(time_uni, time, magnitude)
        
        # FFT
        fft_values = fft.fft(mag_uni - np.mean(mag_uni))
        freqs = fft.fftfreq(len(time_uni), d=(time_uni[1] - time_uni[0]))
        
        # Основна частота
        positive_mask = (freqs > 0) & (freqs < 1)
        periods_fft = 1/freqs[positive_mask]
        power_fft = np.abs(fft_values[positive_mask])
        
        period_mask = (periods_fft >= period_range_hours[0]) & (periods_fft <= period_range_hours[1])
        if np.any(period_mask):
            period_fft = periods_fft[period_mask][np.argmax(power_fft[period_mask])]
            print(f"FFT: {period_fft:.4f} годин")
        else:
            period_fft = period_ls
    except:
        period_fft = period_ls
    
    # Метод 3: Дисперсійний аналіз
    best_period_var = period_ls
    best_variance = np.inf
    
    test_periods = np.linspace(period_range_hours[0], period_range_hours[1], 200)
    for period_test in test_periods:
        period_days = period_test / 24
        phased = (time / period_days) % 1
        
        sort_idx = np.argsort(phased)
        mag_sorted = magnitude[sort_idx]
        
        variance = np.var(mag_sorted)
        
        if variance < best_variance:
            best_variance = variance
            best_period_var = period_test
    
    print(f"Мінімізація дисперсії: {best_period_var:.4f} годин")
    
    # Комбінуємо результати
    final_period = period_ls # бо ls найточніший із методів
    
    # Перевіряємо якість періоду
    quality = evaluate_period_quality(time, magnitude, final_period / 24)
    print(f"Якість періоду: {quality:.3f}")
    
    return final_period / 24  # назад в дні, аби промалювати гарний графік

def evaluate_period_quality(time, magnitude, period_days):

    phased = (time / period_days) % 1
    
    # Розбиваємо на 10 фаз
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    
    bin_means = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (phased >= bins[i]) & (phased < bins[i+1])
        count = np.sum(mask)
        bin_counts.append(count)
        if count > 0:
            bin_means.append(np.mean(magnitude[mask]))
    
    # Критерій 1: рівномірність розподілу точок
    uniformity = np.min(bin_counts) / np.mean(bin_counts) if bin_counts else 0
    
    # Критерій 2: амплітуда кривої
    if len(bin_means) >= 4:
        #amplitude = np.max(bin_means) - np.min(bin_means)
        amplitude = np.max(magnitude) - np.min(magnitude)
    else:
        amplitude = 50
    
    # Критерій 3: гладкість кривої
    if len(bin_means) > 1:
        smoothness = 1 / (1 + np.mean(np.abs(np.diff(bin_means))))
    else:
        smoothness = 0
    
    # Комбінована оцінка
    quality_score = (uniformity + amplitude * 10 + smoothness) / 3
    return min(quality_score, 1.0)

def fourier_model_fit(time, magnitude, period_days, n_terms=3):

    phased_time = (time / period_days) % 1
    
    def fourier_series(phi, *coeffs):
        result = coeffs[0]  # a0
        for n in range(1, n_terms + 1):
            a_n = coeffs[2*n - 1]
            b_n = coeffs[2*n]
            result += a_n * np.cos(2 * np.pi * n * phi)
            result += b_n * np.sin(2 * np.pi * n * phi)
        return result
    
    # П.У.
    initial_guess = [np.mean(magnitude)]
    for n in range(1, n_terms + 1):
        initial_guess.extend([0.05, 0.05])
    
    try:
        popt, _ = optimize.curve_fit(
            fourier_series, phased_time, magnitude, p0=initial_guess,
            maxfev=5000
        )
        return popt
    except:
        return initial_guess

# =============================================================================
# =============================================================================
# ЗБІР ОТРИМАНОГО
# =============================================================================
# =============================================================================

# Комбінація даних з усіх фільтрів
combined_data = []
for filter_code in ['zg', 'zr', 'zi']:
    filter_data = synodic_data[synodic_data['filtercode'] == filter_code].copy()
    if len(filter_data) > 0:
        combined_data.append(filter_data)

if combined_data:
    all_data = pd.concat(combined_data, ignore_index=True)
    all_data = all_data.sort_values('obsjd')
else:
    all_data = synodic_data

print(f"Використано точок: {len(all_data)}")
print(f"Фільтри: {list(all_data['filtercode'].unique())}")

# Часовий ряд
time = all_data['obsjd'].values - all_data['obsjd'].min()
magnitude = all_data['H_reduced'].values

# Період
period_days = advanced_period_search(time, magnitude)
period_hours = period_days * 24

print(f"РЕЗУЛЬТАТ: Період обертання = {period_hours:.4f} годин")

# Апроксимування кривої блиску
fourier_params = fourier_model_fit(time, magnitude, period_days, n_terms=3)

# Візуалізація
print("\n4. Побудова графіків...")

# Функція ряду Фур'є для візуалізації
def fourier_series_function(phi, *coeffs):
    result = coeffs[0]
    for n in range(1, 3):  # 3 члени
        a_n = coeffs[2*n - 1]
        b_n = coeffs[2*n]
        result += a_n * np.cos(2 * np.pi * n * phi)
        result += b_n * np.sin(2 * np.pi * n * phi)
    return result

phased_time = (time / period_days) % 1

phi_dense = np.linspace(0, 1, 500)
magnitude_model = fourier_series_function(phi_dense, *fourier_params)

# Кольори для фільтрів
filter_colors = {'zg': 'green', 'zr': 'red', 'zi': 'blue'}
filter_labels = {'zg': 'g-фільтр', 'zr': 'r-фільтр', 'zi': 'i-фільтр'}

#====================================================
#====================================================
#ГРАФІКИ, СТАТИСТИКА, ТОЧНІСТЬ
#====================================================
#====================================================

# Створення графіків
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Графік 1: Фазова крива
for filter_code in filter_colors.keys():
    mask = all_data['filtercode'] == filter_code
    if np.sum(mask) > 0:
        ax1.scatter(phased_time[mask], magnitude[mask], 
                   alpha=0.7, s=40, color=filter_colors[filter_code],
                   label=filter_labels[filter_code], marker='o')

ax1.plot(phi_dense, magnitude_model, 'k-', linewidth=3, label='Апроксимація Фур\'є')
ax1.set_xlabel('Фаза', fontsize=12)
ax1.set_ylabel('Приведена величина H', fontsize=12)
ax1.set_title(f'Фазова крива блиску\nПеріод: {period_hours:.4f} годин', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.invert_yaxis()

# Графік 2: Часовий ряд з періодичною моделлю
time_hours = time * 24

for filter_code in filter_colors.keys():
    mask = all_data['filtercode'] == filter_code
    if np.sum(mask) > 0:
        ax2.scatter(time_hours[mask], magnitude[mask], 
                   alpha=0.7, s=40, color=filter_colors[filter_code],
                   label=filter_labels[filter_code], marker='o')
# + пперіодична модель
time_model_dense = np.linspace(0, time_hours.max() * 1.1, 1000)
phased_model = (time_model_dense / period_hours) % 1
magnitude_time_model = fourier_series_function(phased_model, *fourier_params)

ax2.plot(time_model_dense, magnitude_time_model, 'k-', linewidth=2, 
         label='Періодична модель', alpha=0.8)
ax2.set_xlabel('Час (години)', fontsize=12)
ax2.set_ylabel('Приведена величина H', fontsize=12)
ax2.set_title('Часовий ряд спостережень', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.invert_yaxis()


plt.tight_layout()
plt.show()

# =============================================================================
# =============================================================================
# ПЕРІОДОГРАМИ
# =============================================================================
# =============================================================================

fig_period, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Графік 1: Періодограма Ломба-Скаргла
print(" - Періодограма Ломба-Скаргла...")

frequencies_ls = np.linspace(1/11.0, 1/1.0, 5000)  # 1-11 годин у мінус 1 степені
power_ls = LombScargle(time, magnitude).power(frequencies_ls)
periods_ls = 1/frequencies_ls 

ax1.plot(periods_ls, power_ls, 'b-', linewidth=1.5, label='Періодограма Ломба-Скаргла')
ax1.axvline(period_hours, color='red', linestyle='--', linewidth=2, 
           label=f'Вибраний період: {period_hours:.4f} год')
ax1.set_xlabel('Період (години)', fontsize=12)
ax1.set_ylabel('Потужність', fontsize=12)
ax1.set_title('Періодограма Ломба-Скаргла', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1, 4)  
ax1.set_ylim(0, 1) 

# Найвищі піки
peaks_ls = []
for i in range(1, len(power_ls)-1):
    if power_ls[i] > power_ls[i-1] and power_ls[i] > power_ls[i+1] and power_ls[i] > 0.1:
        peaks_ls.append((periods_ls[i], power_ls[i]))

if peaks_ls:
    # Топ-3 за потужністтю
    peaks_ls.sort(key=lambda x: x[1], reverse=True)
    for i, (period, pwr) in enumerate(peaks_ls[:3]):
        ax1.annotate(f'{period:.2f} год', xy=(period, pwr), xytext=(5, 5),
                    textcoords='offset points', fontsize=9, color='darkgreen')

# Графік 2: Спектр потужності Фур'є
print(" - Спектр потужності ФФТ...")

n_points = min(312, len(time))
time_uni = np.linspace(time.min(), time.max(), n_points)
mag_uni = np.interp(time_uni, time, magnitude)

fft_values = fft.fft(mag_uni - np.mean(mag_uni))
freqs = fft.fftfreq(len(time_uni), d=(time_uni[1] - time_uni[0]))

positive_mask = (freqs > 0) & (freqs < 2.0)  # До 2 год у мінус 1 степені
periods_fft = 1/freqs[positive_mask]
power_fft = np.abs(fft_values[positive_mask])

# Обмеження діапазону періодів для порівняння з Ломб-Скарглом
period_mask = (periods_fft >= 1) & (periods_fft <= 11)
periods_fft_filtered = periods_fft[period_mask]
power_fft_filtered = power_fft[period_mask]

ax2.plot(periods_fft_filtered, power_fft_filtered, 'g-', linewidth=1.5, 
         label='Спектр потужності ФФТ')
ax2.axvline(period_hours, color='red', linestyle='--', linewidth=2,
           label=f'Вибраний період: {period_hours:.4f} год')
ax2.set_xlabel('Період (години)', fontsize=12)
ax2.set_ylabel('Амплітуда', fontsize=12)
ax2.set_title('Спектр потужності Швидкого Перетворення Фур\'є', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(2, 5)

# Найвищі піки для ФФТ
peaks_fft = []
for i in range(1, len(power_fft_filtered)-1):
    if (power_fft_filtered[i] > power_fft_filtered[i-1] and 
        power_fft_filtered[i] > power_fft_filtered[i+1] and 
        power_fft_filtered[i] > np.mean(power_fft_filtered)):
        peaks_fft.append((periods_fft_filtered[i], power_fft_filtered[i]))

if peaks_fft:
    peaks_fft.sort(key=lambda x: x[1], reverse=True)
    for i, (period, pwr) in enumerate(peaks_fft[:3]):
        ax2.annotate(f'{period:.2f} год', xy=(period, pwr), xytext=(5, 5),
                    textcoords='offset points', fontsize=9, color='darkred')

plt.tight_layout()
plt.show()

# Інформація про піки
print("\nТоп-3 піки в періодограмі Ломба-Скаргла:")
for i, (period, power) in enumerate(peaks_ls[:3], 1):
    print(f"  {i}. {period:.4f} годин (потужність: {power:.4f})")

print("\nТоп-3 піки в спектрі ФФТ:")
for i, (period, power) in enumerate(peaks_fft[:3], 1):
    print(f"  {i}. {period:.4f} годин (амплітуда: {power:.4f})")

# Збереження
fig_period.savefig('periodograms.png', dpi=300, bbox_inches='tight')
print("\nПеріодограми збережено у файл: periodograms.png")
#==============================================================================
#==============================================================================

# Статистика + похибки

predicted_magnitude = fourier_series_function(phased_time, *fourier_params)
residuals = magnitude - predicted_magnitude

rms_error = np.sqrt(np.mean(residuals**2))
amplitude = np.max(magnitude_model) - np.min(magnitude_model)

print(f"RMS похибка: {rms_error:.4f} mag")
print(f"Амплітуда кривої блиску: {amplitude:.4f} mag")
print(f"Відношення сигнал/шум: {amplitude/rms_error:.2f}")

# Статистика по фільтрам
print("\nЯкість даних по фільтрам:")
for filter_code in all_data['filtercode'].unique():
    mask = all_data['filtercode'] == filter_code
    filter_rms = np.sqrt(np.mean(residuals[mask]**2))
    print(f"  {filter_code}: {np.sum(mask)} точок, RMS: {filter_rms:.4f} mag")

# Оцінка похибки періоду
if amplitude > 0:
    period_error_hours = 0.15 * period_hours * (rms_error / amplitude)
else:
    period_error_hours = 0.1 * period_hours

print(f"Період: {period_hours:.4f} ± {period_error_hours:.4f} годин")
print(f"Відносна похибка: {period_error_hours/period_hours*100:.1f}%")

# 7. Збереження результатів
print("\n7. Збереження результатів...")

results = {
    'period_hours': [period_hours],
    'period_error_hours': [period_error_hours],
    'amplitude_mag': [amplitude],
    'rms_error': [rms_error],
    'signal_to_noise': [amplitude/rms_error],
    'n_points_total': [len(all_data)],
}

# Додаємо кількість точок по фільтрам
for filter_code in ['zg', 'zr', 'zi']:
    count = len(all_data[all_data['filtercode'] == filter_code])
    results[f'n_points_{filter_code}'] = [count]

results_df = pd.DataFrame(results)
print("\nРезультати аналізу:")
print(results_df.to_string(index=False))
# Збереження у файл
output_file = 'asteroid_rotation_period_results.csv'
results_df.to_csv(output_file, index=False)
print(f"\nРезультати збережено у: {output_file}")
