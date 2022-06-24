import csv, argparse, os, re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class Wiki103():
    '''
    keyword 뽑아내고 pickle로 파일 저장하기
    하나의 csv 파일에서 단락 & 단락 키워드 & 문장 & 문장 키워드, 4개 list 파일 pickle로 저장
    train 시에는  전처리된 text-keyword pickle 파일을 불러와서, dataset을 만들어 사용

    preprocessing : 핵심 keyword
    preprocessing_stopword : stopword 제외 모든 keyword

    미리 만들어져 있는 파일이 있는지 check 후 필요한 파일을 생성.
    단락(문장) 파일 먼저 체크, 그 다음 키워드 파일 체크.

    단락(문장)과 키워드의 list 개수가 같아야 한다.
    키워드 생성 후 빈 키워드의 경우가 있으므로 체크한다.
    빈 키워드와 매칭되는 단락(문장)이라면 빈 키워드와 단락(문장)을 삭제한다.
    check_is_empty 함수
    '''

    def __init__(self, args):
        self.data_dir = args.data_dir
        self.text_max_length = args.max_length
        self.text_min_length = args.min_length
        self.out_path = args.out_path

    # csv file read
    def read_csv(self, flag):
        if flag == 'train':
            file_name = 'wikitext-103-train.csv'
        elif flag == 'test':
            file_name = 'wikitext-103-test.csv'
        elif flag == 'valid':
            file_name = 'wikitext-103-valid.csv'
        else:
            raise AssertionError('Incorrect Flag value')

        path = self.data_dir + file_name
        with open(path, encoding='UTF-8') as f:
            data_reader = csv.reader(f)
            csv_data = [row for row in data_reader]
        return self.raw_csv_to_list(csv_data)

    # csv to list
    # 미리 설정한 minimum length 이하 text 제외
    def raw_csv_to_list(self, csv_data):
        para_list = []
        for lines in csv_data:
            paragraph = ''
            for line in lines:
                paragraph += line
            if len(paragraph) >= self.text_min_length:
                para_list.append(paragraph)
        return para_list

    # csv 파일은 단락으로 data가 나눠져 있음. 전처리를 두 가지 버전으로 진행. 단락 & 한 문장
    # 단락 -> 한 문장으로 변경하기
    def one_sentence(self, para_list):
        li = []
        for para in para_list:
            sentences = para.split('.')
            for sentence in sentences:
                if sentence and len(sentence) >= 50:
                    li.append(sentence.strip() + '.')
        return li

    def csv_to_list(self, flag):
        para_list = self.read_csv(flag)
        return self.one_sentence(para_list)

    # 전처리 된 데이터 파일로 저장
    def pickle_save(self, path, file_name, file):
        if not os.path.isdir(path):
            os.makedirs(path)
        with open(f'{path}{file_name}.txt', 'wb') as f:
            pickle.dump(file, f)

    def preprocessing_stopword(self):
        flags = ['train', 'test', 'valid']
        out_path = self.out_path
        for flag in flags:
            para_list = self.csv_to_list(flag)
            para_list, keyword_list = self.gen_keyword_list_stopword(para_list)
            result = {'input': keyword_list, 'output': para_list}
            self.pickle_save(out_path, f'{flag}', result)
            print(f'{flag}.txt creation completed!!!')

    def check_is_empty(self, para_list, keyword_list):
        para_li = []
        keyword_li = []
        for i in range(len(para_list)):
            if keyword_list[i]:
                para_li.append(para_list[i])
                keyword_li.append(keyword_list[i])
        return para_li, keyword_li

    '''
    stopword(unk 추가) list 준비
    text에 문자나 숫자를 제외한 것들 제거
    text를 단어로 나누고, stopword와 비교하여 아니면 keyword에 추가
    '''
    def gen_keyword_list_stopword(self, para_list):
        stop_words = set(stopwords.words('english'))
        # add stop word
        stopword = ["unk"]
        for i in stopword:
            stop_words.add(i)
        keyword_list = []
        for para in para_list:
            line = " ".join(re.findall("[a-zA-Z0-9]+", para))
            word_tokens = word_tokenize(line)
            result = []
            for w in word_tokens:
                if w not in stop_words:
                    result.append(w)
            keyword_list.append(result)
        return self.check_is_empty(para_list, keyword_list)

'''
 self.data_dir = args.data_dir
        self.text_max_length = args.text_max_length
        self.text_min_length = args.text_min_length
        self.out_path = args.out_path
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset/source/', required=False)
    parser.add_argument('--max_length', type=int, default=500, required=False)
    parser.add_argument('--min_length', type=int, default=100, required=False)
    parser.add_argument('--out_path', type=str, default='dataset/', required=False)
    args = parser.parse_args()

    nltk.download('stopwords')
    nltk.download('punkt')

    wiki103 = Wiki103(args)
    wiki103.preprocessing_stopword()


