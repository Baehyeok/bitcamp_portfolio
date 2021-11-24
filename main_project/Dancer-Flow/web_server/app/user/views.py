from django.views.generic import TemplateView
from django.views import View
from django.template import context, loader
from django.http import HttpResponse
from django.http.response import JsonResponse
from django.shortcuts import render, redirect

from user.firebase_model import User

from validate_email import validate_email
from django.contrib.auth.hashers import make_password, check_password #비밀번호 암호화 / 패스워드 체크(db에있는거와 일치성확인)

import json
from server.settings import DB

# Create your views here.



class UserView(TemplateView):

    def get(self, req, user_id=''):
        
        print(req.session.get('user'))
        if user_id == '':
        # if req.session.get('user'):
            user_id = req.session.get('user')
        print(user_id)

        result = User.isExistsID(user_id)
        if result:
            user = result.to_dict()
            # print(user)
            play_list = []
            play_doc = DB.collection('Play').stream()
            for play in play_doc:
                data = play.to_dict()
                date = data['date']
                if data['user'].id == user_id:
                    play_list.append({
                        **data, 
                        'pid' : play.id,
                        'date' : '%04d-%02d-%02d %02d:%02d:%02d' % (date.year, date.month, date.day, date.hour+9, date.minute, date.second),
                    })
            # print(play_list)
            context = {
                'user': user,
                'play_list' : play_list
            }

            return render(req, 'user_page.html', context)
        else:
            render('error')

class EmailValidationView(View):
    def post(self, request):
        data=json.loads(request.body) # 데이터를 먼저 불러옴
        email = data['email']
        print(data)
        if not validate_email(email):
            return JsonResponse({'email_error':'Email is invalid'}, status=400) #// 400 Bad request
        if User.isExistsEmail(email) :
            return JsonResponse({'email_error':'Sorry email in use, choose another one.'}, status=409) # resource is confliting with the one already have
        return JsonResponse({'email_valid': True})


class UsernameValidationView(View):
    def post(self, request):
        data=json.loads(request.body) # 데이터를 먼저 불러옴
        username = data['username']
        print(data)
        if not str(username).isalnum():
            return JsonResponse({'username_error':'username should only contain alphanumeric number'}, status=400) #// 400 Bad request
        if User.isExistsUserName(username)       :
            return JsonResponse({'username_error':'Sorry username in use, choose another one.'}, status=409) # resource is confliting with the one already have
        return JsonResponse({'username_valid': True})



def register(request):
    if request.method == "GET":
        return render(request, 'register.html')

    elif request.method == "POST":

        username = request.POST.get('username',None)
        email = request.POST.get('email', None)
        password = request.POST.get('password')
        re_password = request.POST.get('re_password')

        if password != re_password :
            return render(request, 'register.html')

        User(username=username, email=email, password=make_password(password)).register()
        return redirect('/user/login')


def login(request):
    

    if request.method == "GET" :
        return render(request, 'login.html')

    elif request.method == "POST":
        login_email = request.POST.get('email', None)
        login_password = request.POST.get('password', None)

        response_data = {}
        if not (login_email and login_password):
            response_data['error']="Enter your email and password."
        else : 
            result = User.isExistsEmail(login_email)    

            if result:
                user = result.to_dict()
                if check_password(login_password, user['password']):
                    request.session['user'] = result.id
                    return redirect('/')
                else:
                    response_data['error'] = "Please check your password and try again."
            else:
                response_data['error'] = 'This email address is not available.'

        return render(request, 'login.html',response_data)

    
def logout(request):
    if request.session.get('user'):
        del(request.session['user'])
    return redirect('/')


