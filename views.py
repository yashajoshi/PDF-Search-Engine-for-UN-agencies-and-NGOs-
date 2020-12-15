from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponseRedirect
from django.views import View
from django.views.generic.edit import CreateView, UpdateView, DeleteView
from django.views.generic import TemplateView
from django.urls import reverse_lazy
from django.http import HttpResponse
from django.template.loader import render_to_string

from web.models import Task, Project

# PDF Parsing
import pandas as pd
import numpy as np
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import nltk
from nltk.corpus import stopwords
import gensim
from nltk.tokenize import word_tokenize
import scipy


#class MainView(LoginRequiredMixin, View) :
class MainView(TemplateView) :
    def get(self, request):
        return render(request, 'web/home.html')#, ctx)

class ProjectsView(TemplateView) :
    template_name = "web/projects.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['project_list'] = Project.objects.all()

        if context['project_list'] and context['project_list'] == 'word2vec' :
            template_name = "web/project_word2vec.html"

        return context

class word2vecView(TemplateView) :
    def get(self, request):
        if request.method == 'GET':
            # Debug
            print(request.GET)
            return render(request, "web/project_word2vec.html")#, context)
    def post(self, request, **kwargs):
        if request.method == 'POST':
            # Debug
            print(request.POST)

            context = {
                'search_query': "",
                'search_results': [],
            }
            query = request.POST['search-query']
            context['search_query'] = query
            
            df = pd.read_csv('model_final.csv')
           
            df.summary = df.summary.astype(str)
            lowered_summary = [k.lower() for k in df.summary]
            df['summary_cleaned'] = lowered_summary

            replaced = []
            for i in df.summary_cleaned:
                i = re.sub(r"[^ a-zA-Z0-9]+", "", i)
                replaced.append(i)

            df['summary_cleaned'] = replaced
            stop = stopwords.words('english')
            splitted = [k.split() for k in df.summary_cleaned]

            new_words = []
            for k in splitted:
                instance = []
                for l in k:
                    if l not in stop:
                        instance.append(l)
                new_words.append(instance)
            
            joined = []
            for i in new_words:
                sentences = " ".join(i)
                joined.append(sentences)
            
            df['summary_cleaned'] = joined

            path = 'GoogleNews-vectors-negative300-SLIM.bin'
            w2vec = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

            def get_embeddings(word):
                if word in w2vec.vocab:
                    return w2vec[word]
                else:
                    return np.zeros(300)
            
            out_dict = {}

            docs = df['summary_cleaned'].values.tolist()

            for sent in docs:
                average_vector = np.mean(np.array([get_embeddings(word) for word in nltk.word_tokenize(sent)]),axis=0)
                d = {sent:average_vector}
                out_dict.update(d)
            
            def get_similarity(query, doc):
                cos_sim = np.dot(query, doc)/(np.linalg.norm(query)*np.linalg.norm(doc))
                return cos_sim
            
            def rank_text(query):
                query_vector = np.mean(np.array([get_embeddings(word) for word in nltk.word_tokenize(query)]),axis=0)
                rank = []
                for k,v in out_dict.items():
                    rank.append((k,get_similarity(query_vector, v)))
                rank = sorted(rank, key=lambda x:x[1], reverse=True)
                return rank
            
            top_10 = rank_text(query)[:10]

            for i in top_10:

                # Create a new PDF dictionary and add it to the list of search results
                pdf = {
                    "title": str(df[(df.summary_cleaned == i[0])].Title.values[0]),
                    "link": str(df[(df.summary_cleaned == i[0])].URL.values[0]),
                    "summary_short": str(df[(df.summary_cleaned == i[0])].summary.values[0])[:600] + "...", # Truncate summary after 450 characters
                    "summary_full": str(df[(df.summary_cleaned == i[0])].summary.values[0]),
                }
                context['search_results'].append(pdf)
            return render(request, "web/project_word2vec.html", context)



class bm25LView(TemplateView) :
    template_name = "web/projects_bm25L.html"

    def get(self, request):
        if request.method == 'GET':
            # Debug
            print(request.GET)
            
            return render(request, "web/project_bm25L.html")

    def post(self, request, **kwargs):
        if request.method == 'POST':
            # Debug
            print(request.POST)

            context = {
                'search_query': "",
                'search_results': [],
            }
            query = request.POST['search-query']
            context['search_query'] = query
            
            # Prepare dataframe
            #Uncomment in test, comment in prod
            df = pd.read_csv('model_final.csv')

            # Extract summaries from PDFs and queries from query list
            df.summary = df.summary.astype(str)
            lowered_summary = [k.lower() for k in df.summary]
            df['summary_cleaned'] = lowered_summary

            replaced = []
            for i in df.summary_cleaned:
                i = re.sub(r"[^ a-zA-Z0-9]+", "", i)
                replaced.append(i)

            df['summary_cleaned'] = replaced
            stop = stopwords.words('english')
            splitted = [k.split() for k in df.summary_cleaned]

            new_words = []
            for k in splitted:
                instance = []
                for l in k:
                    if l not in stop:
                        instance.append(l)
                new_words.append(instance)
            
            joined = []
            for i in new_words:
                sentences = " ".join(i)
                joined.append(sentences)
            
            df['summary_cleaned'] = joined

            from rank_bm25 import BM25L

            docs = df['summary_cleaned'].values.tolist()
            tokenized_corpus = [doc.split(" ") for doc in docs]
            bm25L = BM25L(tokenized_corpus)
            # queries = [x for x in df_queries.Query]
            tokenized_query = query.split(" ")
            retrieve = bm25L.get_top_n(tokenized_query, docs, n=10)
            # Debug
            print('Query: ' + query + '\n')

            for i in retrieve:

                # Create a new PDF dictionary and add it to the list of search results
                pdf = {
                    "title": str(df[(df.summary_cleaned == i)].Title.values[0]),
                    "link": str(df[(df.summary_cleaned == i)].URL.values[0]),
                    "summary_short": str(df[(df.summary_cleaned == i)].summary.values[0])[:600] + "...", # Truncate summary after 450 characters
                    "summary_full": str(df[(df.summary_cleaned == i)].summary.values[0]),
                }
                context['search_results'].append(pdf)       
            # print(context)         

            return render(request, "web/project_bm25L.html", context)