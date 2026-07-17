from ontoaligner.ontology import AgenticXDDataset
from ontoaligner.aligner.pruner.CandidateExtraction  import prune_candidates, _extract_label


## If the the file is TTL, it is turned into XML since it is not supported by OntoAligner yet
# from helper import to_xml
# to_xml('assets/AXD/DPP/Logs4batch_id 1.ttl',"assets/AXD/DPP/Logs4batch_id 1.xml")
# to_xml('assets/AXD/DPP/Logs4batch_id 2.ttl',"assets/AXD/DPP/Logs4batch_id 2.xml")
'''
task = AgenticXDDataset()
dataset = task.collect(
    source_ontology_path="assets/AXD/DPP/Logs4batch_id 1.xml",
    target_ontology_path="assets/AXD/DPP/Logs4batch_id 2.xml",

)

## Seperating source and target classes/ o props and d props into variables
Source_Ontology_classes = [row for row in dataset['source'] if row['type'] == 'class']
Source_Ontology_object_properties = [row for row in dataset['source'] if row['type'] == 'object property']
Source_Ontology_data_properties = [row for row in dataset['source'] if row['type'] == 'data property']

Target_Ontology_classes = [row for row in dataset['target'] if row['type'] == 'class']
Target_Ontology_object_properties = [row for row in dataset['target'] if row['type'] == 'object property']
Target_Ontology_data_properties = [row for row in dataset['target'] if row['type'] == 'data property']

## Pruning candiates (or RAG)
Candidate_classes = prune_candidates(Source_Ontology_classes,Target_Ontology_classes, threshold=0.5)

for i, j, score in Candidate_classes:
    print(f"{_extract_label(Source_Ontology_classes[i])!r:30s}\
     <-> {_extract_label(Target_Ontology_classes[j])!r:20s} score={score:.3f}")
'''


temp = '''BankCashDistribution
CashDividendAction

BankStockBonusDistribution
StockDividendAction

BankShareSplitAction
StockSplit

BankReverseShareConsolidation
ReverseStockSplit

BankDividendChoiceEvent
DividendOptionAction

BankDividendReinvestmentPlan
DividendReinvestmentAction

BankCapitalReturnDistribution
CapitalDistribution

BankCapitalGainPayout
CapitalGainsDistribution

BankRightsSubscriptionEvent
RightsExerciseEvent

BankBonusRightsDistribution
BonusRightsIssue

BankTenderPurchaseOffer
TenderOffer

BankIssuerRepurchaseOffer
RepurchaseOffer

BankOddLotBuybackProgram
OddLotOffer

BankDutchAuctionBuyback
DutchAuction

BankSecurityConversionEvent
ConversionAction

BankConversionSuspensionNotice
ConversionSuspensionAction

BankPostMergerSecurityExchange
PostMergerSecuritiesExchange

BankBondCouponPayment
InterestPaymentAction

BankPaymentInKindInterest
InterestPaymentInKind

BankPrincipalAndInterestPayment
InterestPaymentWithPrincipal

BankBondMaturityRedemption
RedemptionAtMaturityAction

BankIssuerEarlyCallRedemption
FullCallEarlyRedemptionAction

BankHolderPutRedemption
PutRedemptionAction

BankPartialPrincipalRedemption
PartialRedemptionWithReductionOfNominalValueAction

BankParValueRedenomination
RedenominationAction

BankInterestRateReset
InterestRateAdjustment

BankBondDefaultNotice
BondDefaultAction

BankTradingHaltMessage
TradingStatusSuspendedMessage

BankTradingResumeMessage
TradingStatusActiveMessage

BankDelistingMessage
ListingStatusDelistingMessage

BankWorthlessSecurityNotice
WorthlessSecurityAction
'''.split('\n')
temp = [i for i in temp if i != '']
A = []
B = []
for i in range(0,len(temp),2):
    A.append(temp[i])
    B.append(temp[i+1])

# print(temp)
# Source_Ontology_classes = [row for row in dataset['source'] if row['type'] == 'class']
# Target_Ontology_classes = [row for row in dataset['target'] if row['type'] == 'class']


Candidate_classes = prune_candidates(A,B, threshold=0.25)

for i, j, score in Candidate_classes:
    print(f"{_extract_label(A[i])!r:30s}\
     <-> {_extract_label(B[j])!r:20s} score={score:.3f}")


