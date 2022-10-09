from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def gram_schmidt(A):
    
    (n, m) = A.shape
    
    for i in range(m):
        
        q = A[:, i] # i-th column of A
        
        for j in range(i):
            q = q - np.dot(A[:, j], A[:, i]) * A[:, j]
        
        if np.array_equal(q, np.zeros(q.shape)):
            raise np.linalg.LinAlgError("The column vectors are not linearly independent")
        
        # normalize q
        q = q / np.sqrt(np.dot(q, q))
        
        # write the vector back in the matrix
        A[:, i] = q

# II.1 find the most similar avarge embedding vector t_i,
# II.2 $steps[i] := comp_{\vec{t_i}}{\vec{e}}$
def get_most_cos_sim(emb, topic_avg_emb):
    NUM_TOPICS = len(topic_avg_emb)
    cos_sims = -2*np.ones(NUM_TOPICS)
    for i in range(NUM_TOPICS):
        if topic_avg_emb[i] is not None:
            cos_sims[i] =  cosine_similarity(topic_avg_emb[i], emb).reshape(-1)
    return np.argmax(cos_sims), np.max(cos_sims)

# II.3 repeat steps II.1-2 like Gram-Schmidt (only difference is all $\vec{t_i}$ might not be orthogonal)
def get_topic_dir_steps(emb, topic_avg_emb):
    NUM_TOPICS = len(topic_avg_emb)
    steps = np.zeros(NUM_TOPICS)
    topic_embs = topic_avg_emb.copy()

    for _ in range(len(topic_avg_emb)):
        most_sim_idx, step_sz = get_most_cos_sim(emb, topic_embs)
        steps[most_sim_idx] = step_sz
        emb = emb - step_sz*topic_avg_emb[most_sim_idx]
        topic_embs[most_sim_idx] = None
    return steps

topic_avg_emb_ex = [np.array([1,-1,1]).reshape(1,-1),
                    np.array([1,0,1]).reshape(1,-1)]
for k in range(len(topic_avg_emb_ex)):
    topic_avg_emb_ex[k] = topic_avg_emb_ex[k]/np.linalg.norm(topic_avg_emb_ex[k])

emb_ex = np.array([1,1,2]).reshape(1,-1)
# get_topic_dir_steps(emb_ex, topic_avg_emb_ex)

A = np.concatenate((topic_avg_emb_ex[0], topic_avg_emb_ex[1]), axis=0).T
# gram_schmidt(A)
p_inv = np.linalg.pinv(A)

A = np.array([[1, -1], [2, 2]])
p_inv = np.linalg.pinv(A)
print(A)
print(p_inv)
