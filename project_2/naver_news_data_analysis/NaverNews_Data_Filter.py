import pandas as pd

#2019년도
folder = ['고량주','럼','막걸리','맥주','보드카','소주','와인','위스키','전통주','지역소주']
for a in folder:
    df = pd.read_csv('data/네이버%s뉴스_2019년도.csv'%(str(a))) ## 데이터 불러오기
    df = df.drop_duplicates(subset=None,keep='first',inplace=False) ## 중복값 제거
    df.columns=['num','yyyymmdd','names','title','text'] ## 컬럼이름 변경
    df['yyyymm'] = df['yyyymmdd'].apply(lambda x: x[0:7]) ## yyyymm 컬럼생성
    df['mm'] = df['yyyymmdd'].apply(lambda x: x[5:7]) ## mm 컬럼생성
    df['mmdd'] = df['yyyymmdd'].apply(lambda x: x[5:10]) ## mm 컬럼생성
    df['name'] = a
    ## 2021년 1월 데이터 삭제 ##
    df_drop = df[df['yyyymm']=='2020.01'].index
    df = df.drop(df_drop)
    df_drop = df[df['yyyymm']=='2018.12'].index
    df = df.drop(df_drop)
    ###########################
    df.to_csv('data/Filter_Data/네이버%s뉴스_2019.csv'%(str(a)))

# 2020년도
folder = ['고량주','럼','막걸리','맥주','보드카','소주','와인','위스키','전통주','지역소주']
for a in folder:
    df = pd.read_csv('data/네이버%s뉴스_2020년도.csv'%(str(a))) ## 데이터 불러오기
    df = df.drop_duplicates(subset=None,keep='first',inplace=False) ## 중복값 제거
    df.columns=['num','yyyymmdd','names','title','text'] ## 컬럼이름 변경
    df['yyyymm'] = df['yyyymmdd'].apply(lambda x: x[0:7]) ## yyyymm 컬럼생성
    df['mm'] = df['yyyymmdd'].apply(lambda x: x[5:7]) ## mm 컬럼생성
    df['mmdd'] = df['yyyymmdd'].apply(lambda x: x[5:10]) ## mm 컬럼생성
    df['name'] = a
    ## 2021년 1월 데이터 삭제 ##
    df_drop = df[df['yyyymm']=='2021.01'].index
    df = df.drop(df_drop)
    ###########################
    df.to_csv('data/Filter_Data/네이버%s뉴스_2020.csv'%(str(a)))