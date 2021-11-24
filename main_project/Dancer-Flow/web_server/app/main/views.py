from django.http.response import JsonResponse
from django.views.generic import TemplateView
from django.template import context, loader
from django.http import HttpResponse
from django.shortcuts import render, redirect
from user.firebase_model import User
from numpy import average

from server.settings import DB, BASE_DIR, PROJECT_DIR

import json

class ErrorView(TemplateView):
    template_name = 'error.html'

    def get(self, req):
        return render(req, self.template_name)

class TestView(TemplateView):
    template_name = 'test.html'

    def get(self, req):
        res = render(req, self.template_name)
        res.status_code = 206
        res['Accept-Ranges'] = 'bytes'
        # res['Content-Range'] ='bytes 0-1'
        # res['Content-Length'] ='6'

        return res


class IndexView(TemplateView):
    
    template_name = 'index.html'
    def get(self, req):

        template = loader.get_template(self.template_name)
        context = {'req_id':1}
        return HttpResponse(template.render(context, req))

class StaticView(TemplateView):
    
    def get(self, req, static_name):
        if static_name == 'aboutus':
            return render(req, 'aboutus.html')
        elif static_name == 'introduce':
            return render(req, 'introduce.html')
        elif static_name == 'contactus':
            return render(req, 'contactus.html')
        else:
            return redirect('error')


class CommunityView(TemplateView):
    
    template_name = 'community.html'
    def get(self, req):
        
        play_docs = DB.collection('Play').stream()
        
        user_dict = {
            'faves_list' : [],
            'views_list' : [],
        }
        faves_id_list = []
        if req.session.get('user'):
            user_data = DB.collection(u'User').document(req.session.get('user'))
            user_dict = user_data.get().to_dict()
            for play in user_dict[User.FAVES_FIELD]:
                faves_id_list.append(play.id) 

        data = []
        dict_data = {}
        for doc in play_docs:
            play = doc.to_dict()
            play_user_data = play['user']
            play_user_data = {}
            try:
                play_user_data = play['user'].get().to_dict()

            except Exception as e:
                play_user_data = {
                    'username' : 'Unknown',
                }


            # print(play_user_data['username'])
            date = play['date']
            play['date']  = '%04d-%02d-%02d %02d:%02d:%02d' % (date.year, date.month, date.day, date.hour+9, date.minute, date.second)
            dict_data[date] = {
                'id' : doc.id, 
                'username' : play_user_data['username'],
                **play
            }
        com_data = dict(sorted(dict_data.items(), reverse=True))

        for asd in com_data.values():
            data.append(asd)
        # print(data)
        context = {
            'plays' : data,
            'user' : {**user_dict, 'faves_list' : faves_id_list},
        }
        
        return render(req, self.template_name, context)


    def post(self, req):
        PID = req.POST['pid']
        play_data = DB.collection('Play').document(PID)
        play_dict = play_data.get().to_dict()

        user_data = DB.collection(u'User').document(req.session.get('user'))
        user_dict = user_data.get().to_dict()

        if user_dict[User.FAVES_FIELD]:
            flag = False
            faves_list : list = user_dict[User.FAVES_FIELD]
            faves = play_dict['faves']
            for idx, play in enumerate(user_dict[User.FAVES_FIELD]) :
                if play_data == play:
                    print('일치항목 존재 : ', play_data.id, ' ', play.id)
                    print('fave list 에서 삭제')
                    play_data.update({
                        'faves' : faves - 1 if faves -1 >= 0 else 0
                    })
                    del faves_list[idx]
                    flag = True
                    break
                
            if not flag:
                print('faves list 에 추가')
                play_data.update({
                    'faves' : faves + 1
                })
                faves_list.append(play_data)

            user_data.update({
                User.FAVES_FIELD : faves_list
            })
        else:
            print('faves list 생성')
            user_data.update({
                User.FAVES_FIELD : [play_data]
            })
        
        return JsonResponse({
            'result': 200,
            'type' : req.POST['type'],
        }, json_dumps_params={'ensure_ascii': True})
        

class CommunityVideoView(TemplateView):
    
    template_name = 'community_view.html'

    def get(self, req, play_id):
        # print(req.META )
        # print(req.META['HTTP_REFERER'] )
        play_ref = DB.collection('Play').document(play_id)
        play_data = play_ref.get().to_dict()

        play_data['views']
        play_ref.update({
            'views' : play_data['views'] + 1
        })

        user_data = play_data['user'].get().to_dict()
        print(play_data['user'].id)

        return render(req, self.template_name, {**play_data, 
                                                'user_name' : user_data['email'], 
                                                'views' : play_data['views'] + 1,
                                                'image_path' : user_data['image_path'],
                                                'user_id' : play_data['user'].id})

    def post(self, req):
        pass

class ShareView(TemplateView):

    template_name = 'share.html'
    def get(self, req):
        template = loader.get_template(self.template_name)
        context = {'req_id':1}
        return HttpResponse(template.render(context, req))

    def post(self, req):
        pass


class RankingView(TemplateView):
    
    template_name = 'ranking.html'
    def get(self, req):
        user_doc = DB.collection('User').stream()
        play_doc = DB.collection('Play').stream()
        song_doc = DB.collection('Song').stream()
        
        play_user_data_list = []
        play_song_data_list = []
        user_data_list = []
        song_kpop_data_list = []
        song_pop_data_list = []
        
        play_ranking = {}
        song_ranking = {}
        
        song_count = 0
        upload_time = 0
        favorite = 0
        score = 0
 

        for pdoc in play_doc:
            play_data = pdoc.to_dict()
            play_user_data = play_data['user'].get().to_dict()
            play_user_data_list.append({
                "username" : play_user_data["username"],
                "faves" : play_data["faves"],
                "score" : float(play_data["total_score"])
                })

            play_song_datas = play_data['songs']
            for play_song_data in play_song_datas:
                play_song_data = play_song_data.get().to_dict()
                play_song_data_list.append({
                    'genre' : play_song_data['genre'],
                    'title' : play_song_data['title'],
                    'artist' : play_song_data['artist']
                })

        for udoc in user_doc:
            
            user_data = udoc.to_dict()
            
            for i in play_user_data_list:
                if user_data["username"] == i["username"]:
                    upload_time += 1
                    favorite += i["faves"]
                    score += int(i["score"])

            play_ranking[score] = {
                "image" : user_data['image_path'],
                "username" : user_data['username'],
                "uploadtime" : upload_time,
                "favorite" : favorite,
                "score" : score
            }
            upload_time = 0
            favorite = 0
            score = 0
        for sdoc in song_doc:
            song_data = sdoc.to_dict()
            for a in play_song_data_list:
                if song_data["title"] == a["title"]:
                    song_count += 1
            song_ranking[song_count] = {**song_data}

        pranking = dict(sorted(play_ranking.items(), reverse=True))
        sranking = dict(sorted(song_ranking.items(), reverse=True))
        pnum = 0
        for asd in pranking.values():
            pnum += 1
            asd["num"] = pnum
            user_data_list.append(asd)
        skpopnum = 0
        spopnum = 0
        for asdf in sranking.values():
            if asdf['genre'] == 'kpop':
                skpopnum += 1
                asdf["num"] = skpopnum
                song_kpop_data_list.append(asdf)
            elif asdf['genre'] == 'pop':
                spopnum += 1
                asdf["num"] = spopnum
                song_pop_data_list.append(asdf)
        context = {
            'rank_list': user_data_list,
            'song_kpop_list': song_kpop_data_list,
            'song_pop_list': song_pop_data_list
        }
        return render(req, 'ranking.html', context)

    def post(self, req):
        pass

