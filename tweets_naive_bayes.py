#!/bin/env python3

# I have explained using comments in almost all the functions. The basic approach of the algorithm is to delete few words that do no help in classifier.
#Formatting the words to lower case and removing any special characters or numbers
#Also using the concept of strong words which tell us storngly about the locaiton of the tweet. Like word chicago tells strongly that the tweet could be from chicago but could be wrong.
# So combining such words together so that the count of storng words is even more.
#All this is done in the function get_word_after_formatting
#We can keep on adding more strong words to increase the accuracy of the program


from collections import Counter
import re
import sys 
import os

# this contains final set of locaitons to which the classfier should classify the tweets
global_unique_locations = []

#counter that store number of times each location appeared
global_location_frequency = {}

#All the locaitons in the training set. This doesn't care about duplicacies. Lenght of this list should be equal to the all lines in the training set
global_all_locations = []

#Probability of the each location
global_location_probability = {}

#this contains total number of the words in each location, have key as location and value as the number of the words
global_total_number_of_words_at_location = {}

# Dictionary with key as location and value as the counter.
# Counter contains the count of all the words at that location
global_dictionary_with_frequency= {}

# For any tweet this contains the probability of that tweet in all unique locations
global_probability_of_tweet_at_all_locations = {}


training_file = sys.argv[1]
testing_file = sys.argv[2]
output_file = sys.argv[3]

#While tuning the program on the valiation set I found that the model performs better if we remove these words.
stop_words = ['the','for','and','this','you','with','our','i','a','me','is','that','in','to','by']


# Function to store the probibility of the location
def store_location_probability(all_locations, frequency):
    total_number_of_locations = len(all_locations)
    location_probability = {}
    for location in all_locations:
        location_probability[location] = frequency[location] / total_number_of_locations
    return location_probability

#I analysed and found that few words in the training should be given more emphasis because they are somehow directly related to the location
# Hence merging such words into one so that count of that particular words is more.
# For example if there is word #livinginnyc this word contains nyc hence returning just nyc and similar other cases
#also removing unwanted characters and numbers
def get_word_after_formatting(word):
    
    word = word.lower()
    word = re.sub('[^a-zA-Z]', '', word)
    if word.find('nyc') != -1 or word.find('york') != -1:
        return 'nyc'
    if word.find('toronto') != -1:
        return 'toronto'
    if word.find('boston') != -1:
        return 'boston'
    if word.find('chicago') != -1:
        return 'chicago'
    if word.find('washington') != -1:
        return 'washington'
    if word.find('dc') != -1:
        return 'washington'
    if word.find('sandiego') != -1 or word.find('diego') != -1 :
        return 'sandiego'
    if word.find('houston') != -1:
        return 'houston'
    if word.find('orlando') != -1:
        return 'orlando'
    if word.find('sanfrancisco') != -1 or word.find('francisco') != -1 or word == 'sf':
        return 'sanfrancisco'
    if word.find('atlanta') != -1:
        return 'atlanta'
    if word == 'la':
        return 'los'
    return word
    
#taking the training file and creating the world list
def create_word_list_by_location():
    word_list = {}
    all_locations = []
    with open(training_file, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            splits = line.split()
            location = splits[0]
            all_locations.append(location)
            words = splits[1:]
            if location in word_list:
                 word_list[location] = word_list[location] + [get_word_after_formatting(w) for w in words if get_word_after_formatting(w) != '' and  get_word_after_formatting(w) not in stop_words]
            else:
                 word_list[location] = [get_word_after_formatting(w) for w in words if get_word_after_formatting(w) != '' and get_word_after_formatting(w) not in stop_words]
    return  all_locations, word_list

#for each location creating the count for each word in that location
def create_location_frequency_hash(locations):
    return Counter(locations)

#to get the sum of all the words in the training set
def get_total_words_in_system():
    sum = 0
    for location in global_unique_locations:
        sum = sum + global_total_number_of_words_at_location[location]
    return sum
        
# determine probability of the word given location
def get_probability(word, given_location):
    # I also tried the multivariate verion where you add +1 to the numerator also divide number total number of words in training set
    # p = (m +1) / (words_at_location + total_words)
    # But the below formulat gives the better results
    probability = (global_dictionary_with_frequency[given_location][word]) / ((global_total_number_of_words_at_location[given_location]))
    #not using multivariate probability, assign some probability in case this is zero.
    if probability == 0:
        return 0.0000001
    return probability

#determine probability of a tweet at given location
def get_probability_tweet_at_location(tweet,location):
    words = tweet.split()
    probability = 1
    for word in words:
        probability = probability * get_probability(get_word_after_formatting(word),location)
    return global_location_probability[location] * probability

#calculating the probability of the tweet
def get_probability_tweet(tweet):

    for location in global_unique_locations:
        global_probability_of_tweet_at_all_locations[location] = get_probability_tweet_at_location(tweet,location)
        # Taken from: https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
    return max(global_probability_of_tweet_at_all_locations.keys(), key=(lambda key: global_probability_of_tweet_at_all_locations[key]))

#function to write to the output file
def write_to_output_file(estimated_label, actual_label, tweet):
    with open(output_file, 'a') as output_writer:
        output_writer.write(estimated_label+" "+actual_label+" "+tweet+"\n")

# this function tests the model on the testing set
def validate_model():
    with open(testing_file, encoding='utf-8') as f:
        lines = f.readlines()
        correct_prediction_count = 0
        wrong = 0
        #using to check accuracy, hence commented and not deleted
        #total_predictions = len(lines)
        
        #deleting the file if already exists as i am opening the file in append mode.
        if os.path.exists(output_file):
            os.remove(output_file)
        for line in lines:
            splits = line.split()
            original_location = splits[0]
            line = ' '.join(splits[1:])
            predicted_location = get_probability_tweet(line)
            write_to_output_file(predicted_location,original_location,line)
            if predicted_location == original_location:
                correct_prediction_count = correct_prediction_count + 1
            else:
                wrong = wrong + 1
        #Printing accuracy to test the model, hence commented and not deleted
        #print('accuracy = ', correct_prediction_count/total_predictions)

#function to print the top 5 words from each location
# Taken from https://pymotw.com/2/collections/counter.html
def print_top_words_from_each_location():
    for location in global_unique_locations:
        print(location)
        word_list =  global_dictionary_with_frequency[location]
        print(word_list.most_common(5))


#to get the frequency of each word at each location
def get_frequency_list(table):
                
    for key in table:
        table[key] = Counter(table[key])
    return table

#to calculate the count of words at each location
def get_total_words_location(_frequency_table):
    words_dictionary = {}
    for key in _frequency_table:
        words_dictionary[key] = sum(_frequency_table[key].values())
    return words_dictionary
        

global_all_locations, table = create_word_list_by_location()
global_dictionary_with_frequency = get_frequency_list(table)
global_total_number_of_words_at_location = get_total_words_location(global_dictionary_with_frequency)
global_location_frequency = create_location_frequency_hash(global_all_locations)
global_unique_locations = set(global_dictionary_with_frequency.keys())
global_location_probability = store_location_probability(global_all_locations,global_location_frequency)
validate_model()
print_top_words_from_each_location()
