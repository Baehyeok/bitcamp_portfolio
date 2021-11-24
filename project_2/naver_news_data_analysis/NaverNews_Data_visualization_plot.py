import pandas as pd
import matplotlib.pyplot as plt


## 한번에 그리기 ################################################################################################
plt.rcParams['font.family'] = 'NanumGothic' # 폰트 설정
folder = ['고량주','럼','막걸리','맥주','보드카','소주','와인','위스키','전통주','지역소주']
color = ['#404040','#404040','#BD92DE','#A46608','#26467F','#94D33C','#9C071A','#955900','#BD92DE','#94D33C']
years = [2019,2020]

for year in years:
    for a,c in zip(folder,color):
        df = pd.read_csv('data/Filter_Data/네이버%s뉴스_%s.csv'%(str(a),str(year))) ## 데이터 불러오기
        df_groupby = df.groupby(['yyyymm']).agg('size') ## 월별 size 그룹화
        plt.plot(df_groupby,color=c)
    plt.ylim(0,9000)
    plt.title('주류시장') ## 타이틀 설정
    plt.xlabel('2020년 월별',loc='right') 
    plt.legend(folder,loc='upper left', ncol = 3, fontsize=6, frameon = False)
    plt.savefig('주류 월별_%s_%s.png'%(str(folder),str(year)),dpi=1000)
    plt.show()
################################################################################################################



### 분리(높은거와 낮은거)해서 그리기 #############################################################################
plt.rcParams['font.family'] = 'NanumGothic' # 폰트 설정
folder = ['막걸리','맥주','소주','와인']
color = ['#BD92DE','#A46608','#94D33C','#9C071A']
   
for a,c in zip(folder,color):
    df = pd.read_csv('data/Filter_Data/네이버%s뉴스_2020.csv'%(str(a))) ## 데이터 불러오기
    df_groupby = df.groupby(['yyyymm']).agg('size') ## 월별 size 그룹화
    # df_groupby.plot(color=c)
    plt.plot(df_groupby,color=c)

plt.title('주류시장') ## 타이틀 설정
plt.xlabel('2020년 월별',loc='right') 
plt.legend(['막걸리','맥주','소주','와인'],loc='upper left', ncol = 3, fontsize=6, frameon = False)
plt.savefig('PNGdata/주류 월별_%s_2020.png'%(str(folder)))
plt.show()

plt.rcParams['font.family'] = 'NanumGothic' # 폰트 설정
folder = ['고량주','럼','보드카','위스키','전통주','지역소주']
color = ['#404040','#404040','#26467F','#955900','#BD92DE','#94D33C']
   
for a,c in zip(folder,color):
    df = pd.read_csv('data/Filter_Data/네이버%s뉴스_2020.csv'%(str(a))) ## 데이터 불러오기
    df_groupby = df.groupby(['yyyymm']).agg('size') ## 월별 size 그룹화
    # df_groupby.plot(color=c)
    plt.plot(df_groupby,color=c)

plt.title('주류시장') ## 타이틀 설정
plt.xlabel('2020년 월별',loc='right') 
plt.legend(['고량주','럼','보드카','위스키','전통주','지역소주'],loc='upper left', ncol = 3, fontsize=6, frameon = False)
plt.savefig('PNGdata/주류 월별_%s_2020.png'%(str(folder)))
plt.show()
#################################################################################################################



## 품목별 년도 비교 ##############################################################################################
plt.rcParams['font.family'] = 'NanumGothic' # 폰트 설정
folder = ['고량주','럼','막걸리','맥주','보드카','소주','와인','위스키','전통주','지역소주']
color = ['#404040','#404040','#BD92DE','#A46608','#26467F','#94D33C','#9C071A','#955900','#BD92DE','#94D33C']
for a in folder:

    df = pd.read_csv('data/Filter_Data/네이버%s뉴스_2020.csv'%(str(a))) 
    df2 = pd.read_csv('data/Filter_Data/네이버%s뉴스_2019.csv'%(str(a))) 
    df_groupby = df.groupby(['mm']).agg('size') ## 월별 size 그룹화
    df2_groupby = df2.groupby(['mm']).agg('size')
    
    plt.plot(df_groupby)
    plt.plot(df2_groupby)
    plt.title(a) ## 타이틀 설정
    plt.legend(['2020년도','2019년도']) 
    plt.show()
###################################################################################################################