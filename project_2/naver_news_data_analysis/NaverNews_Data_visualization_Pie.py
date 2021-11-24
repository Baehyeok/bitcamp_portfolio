import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'NanumGothic' # 폰트 설정
folder = ['맥주','와인','소주','막걸리','위스키','전통주','지역소주','고량주','럼','보드카']
folder2 = []
years = ['2019','2020']
explodes = [0,0,0,0,0.05,0.10, 0.10,0.15 ,0.15 ,0.2 ]
wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 5}

for year in years:
    for number in range(12):
        df = pd.DataFrame()
        for a in folder:
            data = pd.read_csv('data/Filter_Data/네이버%s뉴스_%s.csv'%(str(a),str(year)))
            data_groupby = data.groupby(['yyyymm']).agg('size')
            s = data_groupby.iloc[number:(number+1)]
            new_Data = pd.DataFrame(s)
            df = pd.concat([df,new_Data])

        plt.pie(x=list(df[0]),
                labels = folder,
                autopct='%.1f%%',
                counterclock=False, 
                startangle=90, 
                explode=explodes, 
                shadow=False, 
                # subplot=True,
                # facecolor='black',
                wedgeprops=wedgeprops)

        plt.rcParams["figure.figsize"]=(8,8) ## 크기 조정
        plt.xlabel('%s월'%(number+1),loc='right', fontsize =20)
        plt.savefig('2020년%s월_pie_plot'%(str(number+1)))
        # plt.show()
        plt.clf()