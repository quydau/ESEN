from models.graph_constructor import GraphConstructor
from utils.preprocessing import ESENDataPreprocessor
from models.semantic_extraction import SemanticInformationExtraction

# Pipeline
graph_constructor = GraphConstructor(window_size=3)
preprocessor = ESENDataPreprocessor(embedding_dim=300)
semantic_extractor = SemanticInformationExtraction(
    input_dim=300, 
    hidden_dim=300,
    num_steps=5
)

claim = "cut taxes 23 times mayor new york city"
evidences = [
      [
        12705,
        "the new york times site search navigation site navigation site mobile navigation advertisement ny region bloomberg plans to cut new york s property taxes by diane 17 2007 mayor michael r bloomberg is proposing to cut property taxes by roughly 5 percent and eliminate the city sales tax on clothing and footwear as new york enjoys the bounty from its booming economy and real estate market city officials said yesterday mr bloomberg plans to outline his proposals today in his annual address to the city council the property tax cuts which would be in addition to an existing 400 annual",
        "nytimes.com"
      ],
      [
        12706,
        "many of these proposals fail money is going to be coming off the backs of working people and in many cases children de blasio said de blasio outlined the specific impact some of trump s proposed cuts could have on new york city the president s budget proposal the mayor noted would cut more than 600 billion in federal funding to medicaid and the children s health insurance program de blasio said city budget officials project that under those cuts kids in new york city would either lose their health insurance or have their coverage cut back severely it is",
        "politico.com"
      ],
      [
        12707,
        "been proud in cattaraugus county to convene several very successful shared services meetings and would appreciate the governors support to continue doing so warren county board of supervisors chairman ron conover said sharing government services is proven to save taxpayers money and speed up response times in emergencies under governor cuomos leadership new york continues to cut property taxes by reducing the cost of local government and with this proposal new york state demonstrates its commitment to running an efficient government and keeping more money in taxpayers pockets plattsburgh mayor colin read said im incredibly pleased to see that all",
        "ny.gov"
      ]
    ]
# Process sample
graphs = graph_constructor.process_claim_evidence(claim, evidences)
prepared_sample = preprocessor.prepare_sample(graphs)
semantic_reps = semantic_extractor(prepared_sample)

# Output
print(semantic_reps['H_c_sem'].shape)  # [num_claim_nodes, 300]
print(len(semantic_reps['H_e_sem']))   # số evidences
