from django.views.generic.base import RedirectView
from play.firebase_model import Play
from json.encoder import JSONEncoder
from django.http.response import HttpResponseRedirect, JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.generic import TemplateView
from django.views.decorators.csrf import csrf_exempt
import os 
import json
from server.settings import DB, BASE_DIR, PROJECT_DIR
from django.core.files.storage import default_storage

UPLOAD_DIR = 'media/user_videos/'
PLAY_DIR = 'media/play_videos/'
PLAY_PREVIEW_DIR = 'media/play_previews/'
CHUNK_DIR = 'media/chunks/'

class OptionView(TemplateView, RedirectView):
    template_name = 'option.html'

    def read_in_chunks(self, file_object, CHUNK_SIZE):
        while True:
            data = file_object.read(CHUNK_SIZE)
            if not data:
                break
            yield data

    def get(self, req):
    
        if req.session.get('user'):
            print(req)
            print(req.session.get('user'))

            song_ref = DB.collection(u'Song')
            docs = song_ref.stream()
            data = []
            for doc in docs:
                data.append({'id' : doc.id, **doc.to_dict() })
            
            context = {
                'songs' : data,
            }

            return render(req, self.template_name, context)
        else:
            return redirect('login')


    def post(self, req):
        print('option recived')
        print(req.POST)
        print(req.FILES)

        songs_list = []
        songs_data = json.loads(req.POST['songs'])
        for song_id in songs_data:
            data = DB.collection(u'Song').document(song_id)
            songs_list.append(data)

        req.session.get('user')
        user_data = DB.collection(u'User').document(req.session.get('user'))

        # create Play document
        doc_ref = DB.collection(u'Play').document()
        
        UPLOAD_PATH = ''
        if req.POST['upload'] == 'upload':
            file_name_slice = req.FILES['video'].name.split('.')
            file_type = file_name_slice[len(file_name_slice)-1]
            UPLOAD_PATH = os.path.join(UPLOAD_DIR, f'user_{doc_ref.id}.{file_type}')

            CHUNK_SIZE = 1024
                
            with open(req.FILES['video'].temporary_file_path(), 'rb') as tf:
                with open(os.path.join(PROJECT_DIR, UPLOAD_PATH), 'wb+') as uf:
                    for chunk in self.read_in_chunks(tf, CHUNK_SIZE):
                        uf.write(chunk)

        
        play_data = Play(req.POST['mode'], req.POST['upload'], songs_list, user_data, UPLOAD_PATH, )
        doc_ref.set(play_data.to_dict())

        pid = doc_ref.id
        print(pid)

        return HttpResponseRedirect(f'/play/?pid={pid}')


        
class PrePlayView(TemplateView):
    template_name = 'preplay.html'

    def get(self, req):
        return redirect('error')



class PlayView(TemplateView):
    template_name = 'play.html'
    
    def get(self, req):
        
        print('play')
        print(req.GET)

        play_doc = DB.collection(u'Play').document(req.GET['pid']).get()

        if play_doc.exists:
            play_data = play_doc.to_dict()

            if play_data['status'] == 'done':
                return redirect('error')

            song_list = []
            for song_id in play_data['songs']:
                song_list.append(song_id.get().to_dict())
        

            user_data = play_data['user'].get().to_dict()

            context = {
                'pid' : req.GET['pid'],
                'title' : song_list[0]['title'],
                'artist' : song_list[0]['artist'],
                'play_mode' : play_data['option_upload'],
                'video_url' : song_list[0]['file_path'],
                'user_video_url' :  os.path.join('/', play_data['upload_path']),
                'user' : user_data,
            }

            return render(req, self.template_name, context)
        else :
            return redirect('error')

    def post(self, req):
        print('video recived')
        print(req.POST)
        print(req.FILES)

        # if req.POST['type'] == 'play':
        CHUNK_SIZE = 1024
        # Play Video Save
        PLAY_PATH = os.path.join(PLAY_DIR, f"play_{req.POST['pid']}.mp4")
        with default_storage.open(os.path.join(PROJECT_DIR,PLAY_PATH),'wb+') as destination:
            for chunk in req.FILES['play_video'].chunks(CHUNK_SIZE):
                destination.write(chunk)            

        # Play Preview Save
        print(req.POST['preview_path'])
        PREVIEW_PATH = os.path.join(PLAY_PREVIEW_DIR, f"play_preview_{req.POST['pid']}.mp4").replace('\\', '/')
        os.replace(os.path.join(PROJECT_DIR, f"./{req.POST['preview_path']}"), os.path.join(PROJECT_DIR, f"{PREVIEW_PATH}"))
        print('preview_path ', PREVIEW_PATH)

        # Remove Chunk Data
        chunk_path = os.path.join(PROJECT_DIR, CHUNK_DIR)
        chunk_list =  os.listdir(chunk_path)
        for file in chunk_list:
            if file.startswith(f"{req.POST['pid']}_"):
                try:
                    os.remove(os.path.join(chunk_path, file))
                except Exception as e:
                    print('chunk file remove error')
                    print(e)


        play_ref = DB.collection('Play').document(req.POST['pid'])
        play_ref.update({
            'play_path' : os.path.join('/', PLAY_PATH),
            'play_preview_path' : os.path.join('/', PREVIEW_PATH),
            'parts_score' : json.loads(req.POST['parts_score']),
            'total_score' : req.POST['total_score'],
            'status' : 'done',
        })


        return JsonResponse({
            'result': 200,
        }, json_dumps_params={'ensure_ascii': True})



class PreShareView(TemplateView):
    template_name = 'preshare.html'

    @csrf_exempt
    def get(self, req):
        print(req.GET)
        
        return redirect('error')

    @csrf_exempt
    def post(self, req):
        print(req.POST)

        IS_SHARED = False if req.POST['title'] == '' else True
        TITLE = req.POST['title']
        CONTENT = req.POST['content']
        TAGS = req.POST['tags'].split()

        play_ref = DB.collection('Play').document(req.POST['pid'])
        play_ref.update({
            'title' : TITLE if TITLE != '' else '제목이 없습니다.',
            'content' : CONTENT if CONTENT != '' else '내용이 없습니다.',
            'tags' : TAGS,
            'isShared' : IS_SHARED,
            'faves' : 0,
            'views' : 0,
            'status' : 'done',
        })

        return redirect(f'/community/view/{req.POST["pid"]}')



