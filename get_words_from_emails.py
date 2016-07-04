# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 20:01:06 2016

@author: John Enyeart
"""

from random import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from parse_out_email_text import parseOutText

def get_possible_sig_words(data_dict):
    '''This function attemps to build a list of signature words to be
    excluded from email text.
    '''
    sig_words = set()
    for person in data_dict:
        names = person.split(' ')
        # skip "people" like "TOTAL"
        if len(names) < 2:
            continue
        first_name = names[1].lower()
        last_name = names[0].lower()
        sig_words.add(first_name)
        sig_words.add(last_name)
        sig_words.add(first_name[0] + last_name)
        sig_words.add(last_name + first_name[0])
        # this is based on a pattern seen in the text learning mini-project
        nsfname = first_name[0] + last_name[:6] + "nsf"
        sig_words.add(nsfname)
    
    # manually adding these words after inspecting output of tfidf clf
    manual_add = ['greg']
    for word in manual_add:
        sig_words.add(word)
    
    return list(sig_words)

def get_emails(email_addr, sig_words, n_emails, to=True):
    '''get email word data for a person using email address.
    To get emails *to* this person, set to=True,
    To get emails *from* this person, set to=False.
    '''
    emails = []
    addr_dir = 'emails_by_address'
    pre = 'to_' if to else 'from_'
    filename = pre + email_addr + '.txt'
    path = addr_dir + '/' + filename
    with open(path, 'r') as f:
        temp_email_locs = f.readlines()
    # get lines that have valid email locations
    email_locs = []
    for loc in temp_email_locs:
        if 'maildir' in loc:
            maildir_idx = loc.index('maildir')
            # You may need to modify this line if you have emails in a
            # different location or with a different naming format
            new_loc = 'C:/enron/' + loc[maildir_idx:-1].replace('.','_')
            email_locs.append(new_loc)
    # if n_emails is specified as 'all', check all the emails
    # if there aren't n_emails to check, check as many as we can
    if n_emails == 'all' or len(email_locs) < n_emails:
        n_emails = len(email_locs)
    # shuffle the email list so we can randomly choose n_emails
    shuffle(email_locs)
    email_locs = email_locs[:n_emails]
    # get email contents
    for loc in email_locs:
        try:
            email = open(loc, 'r')
        except:
            # just going to skip to the next email if one won't open
            print "\nTried and failed to open %s" % loc
            continue
        text = parseOutText(email)
        # close the file after we got the text out of it
        email.close()
        # remove instances of signature words
        for sig_word in sig_words:
            text = text.replace(sig_word, '')
        emails.append(text)
    
    # return the parsed emails as one big string
    return ' '.join(emails)

def get_important_words(words, labels, print_importants=False):
    important_words = []
    features_train, features_test, labels_train, labels_test = train_test_split(
        words, labels, test_size=0.3, random_state=42)
    v = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    features_train = v.fit_transform(features_train).toarray()
    features_test = v.transform(features_test).toarray()
    clf = DecisionTreeClassifier()
    clf.fit(features_train, labels_train)
    for i, importance in enumerate(clf.feature_importances_):
        if importance > 0.2:
            imp_word = v.get_feature_names()[i]
            important_words.append(imp_word)
            if print_importants:
                print '  Word "%s" has importance of %f.' % (imp_word, importance)
    return important_words

def get_word_features(data_dict, n_from_emails=50, n_to_emails=50, print_important=False):
    '''Gets number of high importance words from emails of people in data_dict.
    Because there are so many emails in the enron set, the default is to
    randomly grab n_from_emails and n_to_emails for each person and process
    those.
    
    To search all emails to or from each individual, set n_from_emails and/or
    n_to_emails = 'all'.
    
    Note: The e-mail data can be downloaded at https://www.cs.cmu.edu/~./enron/
    '''
    sig_words = get_possible_sig_words(data_dict)
    
    # get labels (data_dict[person]['poi']) and words
    labels = []
    to_words = []
    from_words = []
    # get names so we can count high importance words from each person later
    names = []
    for person in data_dict:
        labels.append(data_dict[person]['poi'])
        names.append(person)
        email_addr = data_dict[person]['email_address']
        # if they don't have an email address, go to next person
        if email_addr == 'NaN':
            to_words.append('')
            from_words.append('')
            continue
        # get from words (if any)
        if data_dict[person]['from_messages'] == 'NaN':
            from_words.append('')
        else:
            from_words.append(get_emails(email_addr, sig_words, n_from_emails, to=False))
        # get to words (if any)
        if data_dict[person]['to_messages'] == 'NaN':
            to_words.append('')
        else:
            to_words.append(get_emails(email_addr, sig_words, n_to_emails, to=True))
    
    # get important "to" words
    if print_important:
        print '\nImportant "to" words:'
    important_to_words = get_important_words(to_words, labels, print_important)
    # get important "from" words
    if print_important:
        print '\nImportant "from" words:'
    important_from_words = get_important_words(from_words, labels, print_important)
    
    # finally, count number of important words for each person in data_dict
    for i, person in enumerate(names):
        imp_to_words = 0
        for word in important_to_words:
            imp_to_words += to_words[i].count(word)
        data_dict[person]['important_to'] = imp_to_words
        
        imp_from_words = 0
        for word in important_from_words:
            imp_from_words += from_words[i].count(word)
        data_dict[person]['important_from'] = imp_from_words
    
    return data_dict