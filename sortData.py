import glob
import re
from operator import itemgetter
import random
import pandas as pd
import numpy as np
import pprint
import tensorflow as tf
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
from textaugment import EDA
import statistics
import matplotlib.pyplot as plt
import pickle

# Uncomment lines below the first time running this code to download the necessary NLTK datasets!!!
# nltk.download('wordnet')
# nltk.download('stopwords')

# Get a list of all the .story files that are part of the CNN/DM Dataset.
CNN_DM_FILES = 'path/to/cnn_dm_files/*.story'

# Set these variables
SAMPLE_SIZE = 1000 # The total amount of samples in the dataset
BUCKET_SIZE = 200 # The amount of samples in each individual bucket
TRAINING_PER = 0.9  # Percentage of the sample set treated as training data, the remainder will be the validation set.
SORTING_METHOD = "sorting_method_name" # The name of the sorting method, for user reference only
DATASET_NAME = "cnndm" # The name of the dataset, for user reference only

def load_and_save_files(files_path, file_name):
    """Read all .story files in file path and save as .data file with pickle

        Args:
            files_path: [string] The name of the csv file containing the datapoop
            file_name: [string] The name of the file that will be saved

        Returns:
            Nothing

        """
    all_files = []
    files = get_files(files_path)
    with open(file_name + '.pkl', 'wb') as file_list:
        for file in tqdm(files):
            try:
                file_info = get_file_contents(file)
                all_files.append(file_info)
            except:
                print("Oh no! ERROR OCCURED! Oh well...")
                continue
            if len(all_files) == 10000:
                pickle.dump(all_files, file_list)
                all_files = []

# Uncomment lines as required in the main() function below
def main():
    # load_and_save_files(CNN_DM_FILES, "cnn_all_files")
    # print("Loading list from pickle...")
    with open('../thesis/cnn_dm_all_files.pkl', 'rb') as all_files:
        all_files_with_content = pickle.load(all_files)
    print("Finished loading list")

    # Generate summary statistics for whole dataset
    # gen_sum_stats(all_files_with_content)

    print("Sorting files...")
    # Generate summary statistics for sample size of dataset
    # gen_sum_stats(random.sample(all_files_with_content, SAMPLE_SIZE))

    # When not using EDA, uncomment a line below to use that particular sorting method
    # sorted_files = sort_length(random.sample(all_files_with_content, SAMPLE_SIZE))
    # sorted_files = sort_reduction(random.sample(all_files_with_content, SAMPLE_SIZE))
    # sorted_files = sortCl(random.sample(all_files_with_content, SAMPLE_SIZE))
    # sorted_files = random.sample(all_files_with_content, SAMPLE_SIZE)

    # When using EDA, uncomment lines below
    # augmented_files = augmentEDA(random.sample(all_files_with_content, SAMPLE_SIZE))
    # sorted_files = sortCl(augmented_files)

    # Use code below to create buckets for the One-Pass curriculum
    # create_buckets(sorted_files, BUCKET_SIZE, TRAINING_PER, SORTING_METHOD, DATASET_NAME)
    # print("Finished creating buckets")
    
    # Use code below to create buckets for the Baby-Steps curriculum
    # create_CLBS_buckets(sorted_files, BUCKET_SIZE, TRAINING_PER, SORTING_METHOD, DATASET_NAME)
    # print("Finished creating buckets")

    # Use the code below to prepare data into a .csv file and subsequently transform it to a .tfrecord file that can be used in the PEGASUS model
    # print("preparing data...")
    # prepare_data(sorted_files, "temp_cnn_data.csv")
    # print("Transforming summaries from csv file to tfrecord")
    # csv_to_tfrecords("temp_cnn_data.csv", "Datasets/cnn_sorted_CL_1K_bucket_10_validate.tfrecord")

def get_files(folder_path):
    """Reads all .story files from a folder and prints the number of files in the folder.

        Args:
            folder_path: A path to the folder containing the .story files.

        Returns:
            a list of all the files in the folder.

        """
    files = glob.glob(folder_path)
    print("There are {} files in {} folder.".format(str(len(files)), folder_path))
    return files

def get_file_contents(file_path):
    """Extracts all relevant information from a .story file

        Args:
            file_path: [string] The path to the .story file

        Returns:
            A list [article_len, reduction, article, summary, summary_len]

        """
    file = open(file_path, 'r')
    full_text = file.read()

    # Use regex to take all the text until the first '@highlight' is found which indicates the beginning of the summary
    article = re.search('[\s\S]*?(?=@highlight)', full_text).group()
    # Remove '-LRB- CNN -RRB-' tag
    article_str = article.replace("-LRB- CNN -RRB- -- ", '')
    article = article_str.split()
    # print(article)
    article_len = len(article)

    # Use regex to select all the text after the first '@highlight'.
    summary = re.search('(?<=@highlight)[\s\S]*', full_text).group()
    # Remove the '@highlight' tags
    summary = summary.replace("@highlight",'')
    # Remove the empty lines
    summary_str = ''.join([s for s in summary.strip().splitlines(True) if s.strip()])
    summary = summary_str.split()
    summary_len = len(summary)
    # print(summary)

    reduction = ((len(summary) - len(article)) / len(article)) * -100
    output = [article_len, reduction, article, summary, summary_len, article_str, summary_str]
    # print(output)
    return output

def prepare_data(sorted_list, csv_file_name):
    """Convert a list containing data content to a csv file ready for processing.

        Args:
            sorted_list: [list] list of lists like [[int article_length, float reduction, string article, string summary, ...]]
            csv_file_name: [string] Name of the csv file where the data should be saved, such as "example.csv"

        Returns:
            Nothing

        """
    #Sorted list should be a list of lists like [article_len, reduction, article, summary, summary_len, article_str, summary_str]
    inputs_list = []
    targets_list = []
    for item in tqdm(sorted_list):
        # print(item)
        # inputs_list.append(item[0][2])
        # targets_list.append(item[0][3])
        inputs_list.append(item[5])
        targets_list.append(item[6])

    data_dict = dict(
        inputs=inputs_list,
        targets=targets_list
    )

    df = pd.DataFrame(data_dict)
    header = ["inputs", "targets"]
    df.to_csv(csv_file_name, columns=header, index=False)

def csv_to_tfrecords(csv_file_name, tfrecord_file_name):
    """Read a .csv file and convert contents to a .tfrecords file.

        Args:
            csv_file_name: [string] The name of the csv file containing the data
            tfrecord_file_name: [string] The name of the file the tfrecords should be stored as

        Returns:
            Nothing

        """
    csv = pd.read_csv(csv_file_name).values
    # print(type(csv))
    with tf.io.TFRecordWriter(tfrecord_file_name) as writer:
        for row in tqdm(csv):
            try:
                inputs, targets = row[:-1], row[-1]
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "inputs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[inputs[0].encode('UTF-8')])),
                            "targets": tf.train.Feature(bytes_list=tf.train.BytesList(value=[targets.encode('UTF-8')])),
                        }
                    )
                )
                writer.write(example.SerializeToString())
            except AttributeError as Exception:
                inputs, targets = row[:-1], row[-1]
                print("Oh no! ATTRIBUTE ERROR, oh well...")
                # print(inputs[0])
                # print(targets)

                continue

def gen_sum_stats(file_content_list):
    """Read a list of file contents and calculate and print summary statistics.

        Args:
            file_content_list: [list] file content list: [[int article_length, float reduction, string article, string summary]]

        Returns:
            Nothing

        """
    article_len_list = []
    reduction_list = []
    inputs_list = []
    targets_list = []
    article_words = []
    summ_words = []

    print(file_content_list[0])

    for item in tqdm(file_content_list):
        print(item)
        article_len_list.append(item[0])
        reduction_list.append(item[1])
        inputs_list.append(item[2])
        targets_list.append(item[3])
        article_words.append(len(item[2]))
        summ_words.append(len(item[3]))


    avg_art_len = statistics.mean(article_words)
    avg_red = statistics.mean(reduction_list)

    # plt.hist(reduction_list, bins=100, range=(0,100))
    plt.hist(reduction_list, bins=100, range=(0, 100))
    plt.show()

    print("The average article length is {}".format(avg_art_len))
    print("The minimal article length is {}".format(min(i for i in article_words if i > 0)))
    print("The maximal article length is {}".format(max(article_words)))
    print("The average reduction percentage is {}".format(avg_red))
    print("The minimal reduction percentage is {}".format(min(i for i in reduction_list if i > 0)))
    print("The maximum reduction percentage is {}".format(max(reduction_list)))
    print("The average summary length is {}".format(statistics.mean(summ_words)))
    print("The minimal summary length is {}".format(min(summ_words)))
    print("The maximum summary length is {}".format(max(summ_words)))

    # print(article_len_list)
    # print(reduction_list)
    # print(inputs_list)
    # print(targets_list)

def sort_length(files):
    """Load in files and sort them according to length

        Args:
            files: [list] List of file contents [[article_len, reduction, article, summary, summary_len], ...]

        Returns:
            sorted_files: [list] list of files sorted by lengths

        """
    # for file in tqdm(files):
    #     article_len = file[0]
    #     reduction = file[1]
    #     text = file[2]
    #     summary = file[3]
    #     summ_len = files[4]

    print("There are {} files to be sorted".format(len(files)))
    sorted_files = sorted(files, key=itemgetter(0))
    return sorted_files

def sort_reduction(files):
    """Load in files and sort them according to length

        Args:
            files: [list] List of file contents [[article_len, reduction, article, summary, summary_len], ...]

        Returns:
            sorted_files: [list] list of files sorted by lengths

        """
    # for file in tqdm(files):
    #     article_len = file[0]
    #     reduction = file[1]
    #     text = file[2]
    #     summary = file[3]
    #     summ_len = files[4]

    print("There are {} files to be sorted".format(len(files)))
    sorted_files = sorted(files, key=itemgetter(1))
    return sorted_files

def sortCl(files):
    deletions_list = []
    additions_list = []
    substitutions_list = []
    reordering_list = []
    """Load in files and sort them according to CL strategy

        Args:
            files: [list] List of file contents [article_len, reduction, article, summary, summary_len, article_str, summary_str]

        Returns:
            scored_files: [list] list of files sorted accoring to CL strategy

        """
    scored_files = []
    for file in tqdm(files):
        article_len = file[0]
        reduction = file[1]
        text = file[2]
        summary = file[3]
        summ_len = file[4]
        article_str = file[5]
        summary_str = file[6]
        # print("This file has a text of length {} and a summary of length {}, that is a reduction of {} percent.".format(
        #     str(article_len), str(summ_len), str(round(reduction, 0))))

        # remove punctuation from each word
        import string
        table = str.maketrans('', '', string.punctuation)
        stripped_text = [w.translate(table) for w in text]
        stripped_text_lower = [w.lower() for w in stripped_text]

        stripped_summ = [w.translate(table) for w in summary]
        stripped_summ_lower = [w.lower() for w in stripped_summ]

        # print(stripped_text_lower)

        while '' in stripped_text_lower:
            stripped_text_lower.remove('')

        while '' in stripped_summ_lower:
            stripped_summ_lower.remove('')

        # filter out stop words
        stop_words = ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"]
        text_without_stopwords = [w for w in stripped_text_lower if w not in stop_words]
        summ_without_stopwords = [w for w in stripped_summ_lower if w not in stop_words]

        deletions = 0
        deletions_words = []
        partial_deletions = []
        text_without_del = text_without_stopwords.copy()
        for word in set(text_without_stopwords):
            # print(word)
            current_deletions = text_without_stopwords.count(word) - min(text_without_stopwords.count(word), summ_without_stopwords.count(word))
            deletions += current_deletions
            for i in range(current_deletions):
                deletions_words.append(word)
            index_list = [i for i, x in enumerate(text_without_del) if x == word]

            if current_deletions > 0:
                if len(index_list) == 1:
                    text_without_del.remove(word)
                    # print("The word {} occured in the text once and not in the summary, thus it is deleted.".format(word))
                elif len(index_list) == current_deletions:
                    # print(text_without_del)
                    # print("From the text above, the word {} should have been removed {} times".format(word, current_deletions))
                    text_without_del = list(filter(lambda x: x != word, text_without_del))
                    # print(text_without_del)
                else:
                    partial_deletions.append([word, current_deletions])
                    # print(partial_deletions)
                    # print("There are {} occurences of the word {}.\nOnly {} of these occurences need to be removed.".format(index_list, word, current_deletions))

        for x in partial_deletions:
            word = x[0]
            number_of_deletions = x[1]
            index_list = [i for i, x in enumerate(text_without_del) if x == word]
            index_list_summ = [i for i, x in enumerate(summ_without_stopwords) if x == word]
            # print("DEL: There are {} occurences of the word {}.\nOnly {} of these occurences need to be removed.\nThe summary indexes are {}".format(
            #     index_list, word, number_of_deletions, index_list_summ))


        additions = 0
        additions_words = []
        summ_without_add = summ_without_stopwords.copy()
        duplicate_additions = []
        for word in set(summ_without_stopwords):
            current_additions = summ_without_stopwords.count(word) - min(stripped_text_lower.count(word), summ_without_stopwords.count(word))
            additions += current_additions
            for i in range(current_additions):
                additions_words.append(word)
            index_list = [i for i, x in enumerate(summ_without_add) if x == word]

            if current_additions > 0:
                if len(index_list) == 1:
                    summ_without_add.remove(word)
                    # print(summ_without_add)
                elif len(index_list) == current_additions:
                    summ_without_add = list(filter(lambda x: x != word, summ_without_add))
                else:
                    duplicate_additions.append([word, current_additions])
                    # print("DA: There are {} occurences of the word {}.\nOnly {} of these occurences need to be removed.".format(index_list, word, duplicate_additions))

        delete_from_text = []
        for item in partial_deletions:
            word = item[0]
            index_list = [i for i, x in enumerate(text_without_del) if x == word]
            index_list_summ = [i for i, x in enumerate(summ_without_add) if x == word]

            min_diffs = []
            for index in index_list:
                min_index_diff = np.infty
                for summ_index in index_list_summ:
                    diff = abs(index - summ_index)
                    if diff < min_index_diff:
                        min_index_diff = diff
                min_diffs.append(min_index_diff)

            for i in range(item[1]):
                max_min_diff = max(min_diffs)
                delete_index = min_diffs.index(max_min_diff)
                delete_from_text.append(index_list[delete_index])
                del min_diffs[delete_index]

        # print(set(delete_from_text))
        text_without_del = [i for j, i in enumerate(text_without_del) if j not in set(delete_from_text)]
        # print(text_without_del)

        # print(duplicate_additions)
        delete_from_summ = []
        for item in duplicate_additions:
            word = item[0]
            # print("The duplicate word is: {}".format(word))
            index_list = [i for i, x in enumerate(summ_without_add) if x == word]
            index_list_text = [i for i, x in enumerate(text_without_del) if x == word]
            # print(text_without_del)
            # print(index_list)
            # print(index_list_text)

            min_diffs = []
            for index in index_list:
                max_index_diff = 0
                for text_index in index_list_text:
                    diff = abs(index - text_index)
                    if diff < min_index_diff:
                        min_index_diff = diff
                min_diffs.append(min_index_diff)
                # print("MIN_DIFFS: {}".format(min_diffs))

            for i in range(item[1]):
                # print(min_diffs)
                max_min_diff = max(min_diffs)
                # print(max_min_diff)
                delete_index = min_diffs.index(max_min_diff)
                delete_from_summ.append(index_list[delete_index])
                del min_diffs[delete_index]

        summ_without_add = [i for j, i in enumerate(summ_without_add) if j not in set(delete_from_summ)]

        # print(text_without_del)
        # print(len(text_without_del))
        # print(summ_without_add)
        # print(len(summ_without_add))
        # print(set(text_without_del) == set(summ_without_add))

        three_gram_list_summ = []
        three_gram_list_text = []
        for i in range(len(summ_without_stopwords)-2):
            three_gram_list_summ.append(summ_without_stopwords[i:i+3])

        for i in range(len(text_without_stopwords)-2):
            three_gram_list_text.append(text_without_stopwords[i:i+3])

        reorders = 0
        for gram in three_gram_list_summ:
            reorders = reorders + three_gram_list_summ.count(gram) - min(three_gram_list_text.count(gram), three_gram_list_summ.count(gram))

        # print("3-grams summary:")
        # print(three_gram_list_summ)

        # print("3-grams original text:")
        # print(three_gram_list_text)

        # print("Original text (stripped):")
        # print(stripped_text_lower)
        # print("Summary (stripped):")
        # print(stripped_summ_lower)

        # print(deletions_words)
        # print(additions_words)
        wordnet_lemmatizer = WordNetLemmatizer()

        lemm_deletions = []
        for word in deletions_words:
            lemm_deletions.append(wordnet_lemmatizer.lemmatize(word))

        lemm_additions = []
        for word in additions_words:
            lemm_additions.append(wordnet_lemmatizer.lemmatize(word))

        replacements = set(lemm_deletions).intersection(lemm_additions)

        deletions = deletions - len(replacements)
        additions = additions - len(replacements)

        # print(replacements)
        #
        # print(lemm_deletions)
        # print(lemm_additions)

        # print("Original text (without stopwords):")
        # print(text_without_stopwords)
        # print("Summary (without stopwords):")
        # print(summ_without_stopwords)
        #
        # print("This summary contains {} deletions".format(deletions))
        # print("This summary contains {} additions".format(additions))
        # print("This summary contains {} reorders".format(reorders))
        # print("This summary contains {} replacements".format(len(replacements)))
        score = 0.19 * deletions + 0.53 * reorders + 0.02 * len(replacements) + 0.26 * additions
        additions_list.append(additions)
        deletions_list.append(deletions)
        reordering_list.append(reorders)
        substitutions_list.append(len(replacements))
        # print("This summary has complexity score: {}".format(score))

        file.append(score)
        scored_files.append(file)

    scored_files = sorted(scored_files, key=itemgetter(7))
    print(statistics.mean(additions_list))
    print(statistics.mean(deletions_list))
    print(statistics.mean(reordering_list))
    print(statistics.mean(substitutions_list))
    print(statistics.stdev(additions_list))
    print(statistics.stdev(deletions_list))
    print(statistics.stdev(reordering_list))
    print(statistics.stdev(substitutions_list))

    return(scored_files)

def augmentEDA(files):
    """Load in files and double them with an augmented file for each one

        Args:
            files: [list] List of file contents [article_len, reduction, article, summary, summary_len, article_str, summary_str]

        Returns:
            scored_files: [list] list of augmented files [article_len, reduction, article, summary, summary_len, article_str, summary_str]

        """
    print("There are {} files available before EDA.".format(len(files)))
    augmented_files = []
    index = 0
    np.random.seed(42)
    rand_numbers = np.random.uniform(0, 1, 10000)
    for file in files:
        # print(file)
        index += 1
        length = file[0]
        reduction = file[1]
        text_list = file[2]
        summary_list = file[3]
        summ_len = file[4]
        text = file[5]
        summary = file[6]

        augmented_files.append(file)

        rand = rand_numbers[index-1]

        eda = EDA()

        if rand < 0.25:
            EDA_text = eda.synonym_replacement(text)
            EDA_summary = eda.synonym_replacement(summary)
        elif rand < 0.5:
            EDA_text = eda.random_insertion(text)
            EDA_summary = eda.random_insertion(summary)
        elif rand < 0.75:
            EDA_text = eda.random_swap(text)
            EDA_summary = eda.random_swap(summary)
        else:
            EDA_text = eda.random_deletion(text)
            EDA_summary = eda.random_deletion(summary)

        # EDA_text = EDA_text.split()
        # EDA_summary = EDA_summary.split()

        augmented_files.append([length, reduction, text_list, summary_list, summ_len, EDA_text, EDA_summary])
    print("There are {} augmented examples available after EDA".format(len(augmented_files)))
    return(augmented_files)

def create_buckets(sorted_files, bucket_size, train_perc, sorting_method, dataset):
    number_of_files = len(sorted_files)
    number_of_buckets = int(number_of_files / bucket_size)

    print("There are {} sorted files to be put into {} buckets".format(number_of_files, number_of_buckets))

    for bucket_index in range(number_of_buckets):
        bucket = sorted_files[int(bucket_size*bucket_index):int(bucket_size*(bucket_index+1))]
        print("This bucket has size {}".format(len(bucket)))
        training = bucket[0:int(train_perc*bucket_size)]
        validation = bucket[int(train_perc*bucket_size):bucket_size]
        print("Training has length {}".format(len(training)))
        print("Validation has length {}".format(len(validation)))
        prepare_data(training, "../thesis/temp_train_data.csv")
        prepare_data(validation, "../thesis/temp_val_data.csv")
        csv_to_tfrecords("../thesis/temp_train_data.csv", "Datasets/" + dataset + "_sorted_" + sorting_method + "_" + str(number_of_files) + "_bucket_" + str(bucket_index + 1) + "_train.tfrecord")
        csv_to_tfrecords("../thesis/temp_val_data.csv", "Datasets/" + dataset + "_sorted_" + sorting_method + "_" + str(number_of_files) + "_bucket_" + str(bucket_index + 1) + "_validate.tfrecord")

def create_CLBS_buckets(sorted_files, bucket_size, train_perc, sorting_method, dataset):
    number_of_files = len(sorted_files)
    number_of_buckets = int(number_of_files / bucket_size)

    print("There are {} sorted files to be put into {} buckets".format(number_of_files, number_of_buckets))

    for bucket_index in range(number_of_buckets):
        print(bucket_index)
        bucket = sorted_files[0:int(bucket_size*(bucket_index+1))]
        bs_bucket_size = len(bucket)
        print("This bucket has size {}".format(len(bucket)))
        training = bucket[0:int(train_perc*bs_bucket_size)]
        validation = bucket[int(train_perc*bs_bucket_size):bs_bucket_size]
        print("Training has length {}".format(len(training)))
        print("Validation has length {}".format(len(validation)))
        prepare_data(training, "../thesis/temp_train_data.csv")
        prepare_data(validation, "../thesis/temp_val_data.csv")
        csv_to_tfrecords("../thesis/temp_train_data.csv", "Datasets/" + dataset + "_sorted_" + sorting_method + "_" + str(number_of_files) + "_bucket_" + str(bucket_index + 1) + "_train.tfrecord")
        csv_to_tfrecords("../thesis/temp_val_data.csv", "Datasets/" + dataset + "_sorted_" + sorting_method + "_" + str(number_of_files) + "_bucket_" + str(bucket_index + 1) + "_validate.tfrecord")

main()
