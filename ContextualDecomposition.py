import numpy as np
from scipy.special import expit as sigmoid

"""
Recall:
h_t = beta_t + gamma_t
c_t = beta_ct + gamma_ct

p = SoftMax(W_betaT +W_gammaT)


o_t = sigmoid( W_o*x_t + V_o*h_{t-1} + b_o )
f_t = sigmoid( W_f*x_t + V_f*h_{t-1} + b_f ) 
i_t = sigmoid( W_i*x_t + V_i*h_{t-1} + b_i) 
g_t = tanh(W_g*x_t + V_g*h_{t-1} + b_g)
c_t = f_t * c_{t-1} + i_t * g_t
h_t = o_t * tanh(c_t)
"""

def linearize(a, b, c, activation_fn):
    a_contrib = 0.5 * (activation_fn(a + b + c)  + activation_fn(a + c) - activation_fn(b + c) - activation_fn(c))
    b_contrib = 0.5 * (activation_fn(a + b + c)  - activation_fn(a + c) + activation_fn(b + c) - activation_fn(c))
    return a_contrib, b_contrib, activation_fn(c)

def linearize_tanh(a, b):
    a_contrib = 0.5 * (np.tanh(a) + np.tanh(a + b) - np.tanh(b)) 
    b_contrib = 0.5 * (np.tanh(b) + np.tanh(a + b) - np.tanh(a))
    return a_contrib, b_contrib

def CD(review, model, start, stop):
    weights = model.lstm.state_dict()
    
    W_ii, W_if, W_ig, W_io = np.split(weights['weight_ih_l0'], 4, 0)
    W_hi, W_hf, W_hg, W_ho = np.split(weights['weight_hh_l0'], 4, 0)
    W_out = model.hidden2label.weight.data
    b_i, b_f, b_g, b_o = np.split(weights['bias_ih_l0'].cpu().numpy() + weights['bias_hh_l0'].cpu().numpy(), 4)

    # The second axis is garbage, we remove it. Resulting matrix is of size (#words) x (length of glove vector, 300)
    word_vecs = model.word_embeddings(review)[:,0].data
    
    L = word_vecs.size(0)
    phrase = np.zeros((L, model.hidden_dim))  #phrase contribution
    rest = np.zeros((L, model.hidden_dim))    #rest of the contribution
    phrase_h = np.zeros((L, model.hidden_dim))
    rest_h = np.zeros((L, model.hidden_dim))
    
    #iterate through word_vecs
    for i in range(L):
        if i == 0:
            #there is no prev
            prev_phrase_h = np.zeros(model.hidden_dim)
            prev_rest_h = np.zeros(model.hidden_dim)
        else:
            prev_phrase_h = phrase_h[i-1]
            prev_rest_h = rest_h[i-1]
            
        #calculating o, f, i, g    
        phrase_o = np.dot(W_ho, prev_phrase_h)
        phrase_f = np.dot(W_hf, prev_phrase_h)
        phrase_i = np.dot(W_hi, prev_phrase_h)
        phrase_g = np.dot(W_hg, prev_phrase_h)
        
        rest_o = np.dot(W_ho, prev_rest_h)
        rest_f = np.dot(W_hf, prev_rest_h)
        rest_i = np.dot(W_hi, prev_rest_h)
        rest_g = np.dot(W_hg, prev_rest_h)

        #only modify for range [start, stop]
        if (start <= i) and (i <= stop):
            phrase_o = phrase_o + np.dot(W_io, word_vecs[i])
            phrase_f = phrase_f + np.dot(W_if, word_vecs[i])
            phrase_i = phrase_i + np.dot(W_ii, word_vecs[i])
            phrase_g = phrase_g + np.dot(W_ig, word_vecs[i])
        else:
            rest_o = rest_o + np.dot(W_io, word_vecs[i])
            rest_f = rest_f + np.dot(W_if, word_vecs[i])
            rest_i = rest_i + np.dot(W_ii, word_vecs[i])
            rest_g = rest_g + np.dot(W_ig, word_vecs[i])
        
        #calculate contributions to i, g
        phrase_contrib_i, rest_contrib_i, bias_contrib_i = linearize(phrase_i, rest_i, b_i, sigmoid)
        phrase_contrib_g, rest_contrib_g, bias_contrib_g = linearize(phrase_g, rest_g, b_g, np.tanh)

        phrase[i] = phrase_contrib_i * (phrase_contrib_g + bias_contrib_g) + bias_contrib_i * phrase_contrib_g
        rest[i] = rest_contrib_i * (phrase_contrib_g + rest_contrib_g + bias_contrib_g) + (phrase_contrib_i + bias_contrib_i) * rest_contrib_g

        #add bias for range [start,stop)
        if i >= start and i < stop:
            phrase[i] += bias_contrib_i * bias_contrib_g
        else:
            rest[i] += bias_contrib_i * bias_contrib_g
        
        #When there's a prev, calculate contributions
        if i > 0:
            phrase_contrib_f, rest_contrib_f, bias_contrib_f = linearize(phrase_f, rest_f, b_f, sigmoid)
            phrase[i] += (phrase_contrib_f + bias_contrib_f) * phrase[i-1]
            rest[i] += (phrase_contrib_f + rest_contrib_f + bias_contrib_f) * rest[i-1] + rest_contrib_f * phrase[i-1]

        o = sigmoid(np.dot(W_io, word_vecs[i]) + np.dot(W_ho, prev_phrase_h + prev_rest_h) + b_o)
        phrase_contrib_o, rest_contrib_o, bias_contrib_o = linearize(phrase_o, rest_o, b_o, sigmoid)
        new_phrase_h, new_rest_h = linearize_tanh(phrase[i], rest[i])
        phrase_h[i] = o * new_phrase_h
        rest_h[i] = o * new_rest_h
    
    #calculating final scores
    phrase_scores = np.dot(W_out, phrase_h[L-1])
    rest_scores = np.dot(W_out, rest_h[L-1])

    return phrase_scores, rest_scores

