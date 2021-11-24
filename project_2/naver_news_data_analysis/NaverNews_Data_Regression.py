import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
folder = ['맥주','와인','소주','막걸리','위스키','전통주','지역소주','고량주','럼','보드카']
plt.rcParams['font.family'] = 'NanumGothic'
for a in folder:
    df = pd.DataFrame()
    df1 = pd.read_csv('data/Filter_Data/네이버%s뉴스_2019.csv'%(str(a))) ## 데이터 불러오기
    df2 = pd.read_csv('data/Filter_Data/네이버%s뉴스_2020.csv'%(str(a)))
    df1_groupby = df1.groupby(['mm']).agg('count') ## 월별 size 그룹화
    df2_groupby = df2.groupby(['mm']).agg('count')
    df = pd.concat([df1_groupby,df2_groupby])
    df =df.reset_index('mm')
    x1 = df.mm
    y = df.names
    x = sm.add_constant(x1)
    result = sm.OLS(y,x).fit()
    print(result.summary())
    plt.scatter(x1,y)
    
    plt.xlabel('2019-2020년도_%s_Regression'%(str(a)),loc='right') 
    fig = plt.plot(x1,result.predict(x),lw=4,c='orange' )
    # plt.savefig('PNGdata/regression/2019-2020년도_%s_Regression'%(str(a)))
    plt.show()