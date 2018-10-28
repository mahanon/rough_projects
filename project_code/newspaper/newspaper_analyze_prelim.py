import pandas as pd
import numpy as np
from core_classifier_newspaper import *
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import pickle
import mpld3
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

# Extracted newspaper train (and familiar test) and novel test data, respectively.
gdata_pd = pd.read_pickle('gdata_01_config_01_testrun.pkl')
tdata_pd = pd.read_pickle('tdata_01_config_01_testrun.pkl')


###########
###########
# Core code.

# Wrappr for core_classifier_newspaper script.
# Note that classifier is quite naive, not yet undergoing thorough grid search.
def classify_and_predict(classif=MLPClassifier, train_data_pd=gdata_pd, test_data_pd=tdata_pd, save_pickle = 0):
     # Note that changing classif from MLPClassifier may break current implementation
     # of "test_model," which assumes a generative model.
    
    X = list(train_data_pd['text'])
    y = np.array(train_data_pd['leaning']).T
    
    model_pkg = build_and_evaluate(X,y, classifier=classif)
    # model_pkg contains: leanings_test,leanings_pred_prob,model (see core_classifier_newspaper)

    if save_pickle == 1:
        pickle.dump(model_pkg,'test_mlb_model_pkg.pkl')
        
    return model_pkg

# Evaluate model on test data (novel publications) and generate ROC 
def test_model(y_test,y_pred_prob,tdata_pd, model):
    predictions = model.predict(tdata_pd['text'])
    predictions_prob = model.predict_proba(tdata_pd['text'])
    # assuming order conservative==0, liberal==1
    print(clsr(tdata_pd['leaning']=='liberal', predictions, target_names=['conservative','liberal']))
    #fpr_rf, tpr_rf, _= roc_curve(tdata_pd['leaning']=='liberal', predictions)
    
    fpr_rf, tpr_rf, _= roc_curve(y_test,y_pred_prob[:,1])
    roc_auc = roc_auc_score(y_test,y_pred_prob[:,1])
    print('Liberal AUC, '+str(roc_auc))
    plt.plot(fpr_rf, tpr_rf,label='\'Left\', Familiar Sources, AUC = '+str(roc_auc)[:5],color='green')
    
    fpr_rf, tpr_rf, _= roc_curve(tdata_pd['leaning']=='liberal', predictions_prob[:,1])
    roc_auc = roc_auc_score(tdata_pd['leaning']=='liberal', predictions_prob[:,1])
    print('Liberal AUC, '+str(roc_auc_score(tdata_pd['leaning']=='liberal', predictions_prob[:,1])))
    plt.plot(fpr_rf, tpr_rf,label='\'Left\', Novel Sources, AUC = '+str(roc_auc)[:5],color='blue')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for MLP Prediction of Periodical Text Bias')
    plt.legend(loc='lower right')
    plt.show()
    return None

# Get MDS from TF-IDF matrix (function found in model_pkg)
def compute_mds(tfidf_matrix):
    dist = 1 - cosine_similarity(tfidf_matrix)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist) 
    return pos[:, 0], pos[:, 1]


# Helper for mpld3 plot (tried for fun, modified from public code from Brandon Rose), 
# following function.
class TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    TopToolbar.prototype.draw = function(){
      this.fig.toolbar.draw();

      this.fig.toolbar.toolbar.attr("x", 150);
      this.fig.toolbar.toolbar.attr("y", 400);

      this.fig.toolbar.draw = function() {}
    }
    """
    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}
        
# mpld3 plot of all newspapers plotted in 2D MDS space.
def make_mpld3_mds_plot(xs,ys,data_pd=gdata_pd):

    cluster_colors = {'conservative': '#1b9e77', 'liberal': '#d95f02'}

    # Define new dataframe with MDS coordinates, some things from original DF.
    plot_df = pd.DataFrame(dict(x=xs, y=ys, label=data_pd.loc[:,'leaning'], source=data_pd.loc[:,'source'])) 
    groups = plot_df.groupby('label')

    #css formatting
    css = """
    text.mpld3-text, div.mpld3-tooltip {
    font-family:Arial, Helvetica, sans-serif;
    }

    g.mpld3-xaxis, g.mpld3-yaxis {
            display: none; }

    svg.mpld3-figure {
    margin-left: 0px;}
    """
    # margin as -200px

    fig, ax = plt.subplots(figsize=(14,6))

    for name, group in groups:
        points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18, label=name, mec='none', color=cluster_colors[name])
    ax.set_aspect('auto')
    labels = [i for i in group.source]
    
    # MPLD3 Setup
    tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,voffset=10, hoffset=10, css=css)
    mpld3.plugins.connect(fig, tooltip, TopToolbar())    
    
    # No axes, qualitative plot
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.legend(numpoints=1)

    mpld3.display()

    # export html
    html = mpld3.fig_to_html(fig)
    return

# Make model (using default parameters) or load.
model_pkg = classify_and_predict()
#model_pkg = pickle.load('test_mlb_model_pkg.pkl')

# Evaluate model on test data (novel publications), plot ROC.
test_model(model_pkg[0],model_pkg[1],tdata_pd,model_pkg[2])

# Re-obtain TF-IDF from model, use to compute and plot MDS by publication.
# Note, if environemnt doesn't support plot, use HTML
tfidf_matrix = model[2].named_steps['vectorizer'].transform(gdata_pd['text'])
xs,ys = compute_mds(tfidf_matrix)
mpld3_html = make_mpld3_mds_plot(xs,ys)
