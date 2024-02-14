from fmeval.data_loaders.util import get_dataset
from fmeval.transforms.perturbations import ButterFinger
from fmeval.transforms.summarization_accuracy import SummarizationAccuracy, METEOR_SCORE, ROUGE_SCORE, BERT_SCORE
from fmeval.transforms.util import GeneratePrompt, GetModelResponse, PromptComposer, GenerateDeltaScores, Mean
from fmeval.transforms.transform_pipeline import TransformPipeline
from fmeval.eval_algorithms import DATASET_CONFIGS, XSUM
from test.integration.models.model_runners import sm_model_runner


data_config = DATASET_CONFIGS[XSUM]
ds = get_dataset(data_config, 20)
prompt_composer = PromptComposer("Summarize the following text in one sentence: $feature")

gen_og_prompt = GeneratePrompt(
    prompt_composer=prompt_composer,
    input_keys=["model_input"],
)

get_og_response = GetModelResponse(
    model_runner=sm_model_runner,
    model_input_keys=gen_og_prompt.output_keys,
    model_response_keys=["model_output"],
)

get_perturbed_inputs = ButterFinger(
    input_text_key="model_input",
    perturbation_prob=0.1,
    num_perturbations=3,
)

gen_perturbed_prompt = GeneratePrompt(
    prompt_composer=prompt_composer,
    input_keys=get_perturbed_inputs.output_keys,
)

get_perturbed_responses = GetModelResponse(
    model_runner=sm_model_runner,
    model_input_keys=gen_perturbed_prompt.output_keys,
    model_response_keys=["model_output"]
)

get_og_summ_acc_scores = SummarizationAccuracy(model_output_key=get_og_response.output_keys[0])
bertscore_model_ref = get_og_summ_acc_scores.transforms[BERT_SCORE].bertscore_model_ref

get_perturbed_summ_acc_scores = [
    SummarizationAccuracy(
        model_output_key=perturbed_model_response_key,
        bertscore_model_ref=bertscore_model_ref,
        meteor_score_output_key=f"{METEOR_SCORE}({perturbed_model_response_key})",
        rouge_score_output_key=f"{ROUGE_SCORE}({perturbed_model_response_key})",
        bertscore_output_key=f"{BERT_SCORE}({perturbed_model_response_key})"
    )
    for perturbed_model_response_key in get_perturbed_responses.output_keys
]

original_score_keys = [
    transform.output_keys[0]
    for transform in get_og_summ_acc_scores.transforms.values()
]

perturbed_score_keys = {
    original_score_key: [
        summ_acc.transforms[original_score_key].output_keys[0]
        for summ_acc in get_perturbed_summ_acc_scores
    ]
    for original_score_key in original_score_keys
}

get_delta_scores = [
    GenerateDeltaScores(original_score_key, perturbed_score_keys[original_score_key])
    for original_score_key in original_score_keys
]

get_mean_delta_scores = [
    Mean(delta_score.output_keys)
    for delta_score in get_delta_scores
]

pipeline = TransformPipeline(
    [
        gen_og_prompt,
        get_og_response,
        get_perturbed_inputs,
        gen_perturbed_prompt,
        get_perturbed_responses,
        get_og_summ_acc_scores.pipeline,
        [summ_acc.pipeline for summ_acc in get_perturbed_summ_acc_scores],
        get_delta_scores,
        get_mean_delta_scores
    ]
)

ds = pipeline.execute(ds)
ds.show()

sample = ds.take(1)[0]
for key in sample:
    print(f"{key}\n")

# sample = {'model_input': 'Okinoshima is home to the Okitsu shrine, built in the 17th century to pray for the safety of sailors.\nBefore stepping foot on the island, men must take off their clothes and undergo a cleansing ritual.\nWhen they leave they are not allowed to take away any souvenirs, or disclose details of their visit.\nLong before the shrine was built, Okinoshima was used for rituals involving prayers for oceangoing ships and trade ties with Korean and Chinese people, the Japan Times reports.\nThousands of artefacts brought as gifts from overseas have been found on the island, including gold rings from the Korean Peninsula, it says.\nThe island now welcomes visitors on a single day every year, 27 May, and ancient rules are still observed.\nThe number of visitors is restricted to 200. They must perform ablution rites in the sea, and - most controversially - be male.', 'target_output': "Japan's Okinoshima island, an ancient religious site where women are banned, has been declared a World Heritage site by the UN's cultural body Unesco.", 'GeneratePrompt(model_input)': 'Summarize the following text in one sentence: Okinoshima is home to the Okitsu shrine, built in the 17th century to pray for the safety of sailors.\nBefore stepping foot on the island, men must take off their clothes and undergo a cleansing ritual.\nWhen they leave they are not allowed to take away any souvenirs, or disclose details of their visit.\nLong before the shrine was built, Okinoshima was used for rituals involving prayers for oceangoing ships and trade ties with Korean and Chinese people, the Japan Times reports.\nThousands of artefacts brought as gifts from overseas have been found on the island, including gold rings from the Korean Peninsula, it says.\nThe island now welcomes visitors on a single day every year, 27 May, and ancient rules are still observed.\nThe number of visitors is restricted to 200. They must perform ablution rites in the sea, and - most controversially - be male.', 'GetModelResponse(GeneratePrompt(model_input), model_output)': ' Okinoshima is a small island in Japan', 'model_input.ButterFinger(0)': 'Okinoshika ix home to nhe Okitsi shrine, built in the 17th centjry vo pgay for the rayety mg sailors.\nBefore step[ing foot on the ixland, mwn mhst take off their clothts and undergo a cleansing ritual.\nWfen jhey leave they ase not allowed to take away any solvenirs, or discloss details of dheor visie.\nLong ueford the shxine was built, Okinoshima was ufed for rituals involving prahers fir ocesncoing ahips and trade ties with Korean and Chinqse people, the Japaj Times reports.\nThousqnds ow awtedacts brought as jiyts fgim ovqrseas havr been found ou the island, inglndung gold rinys grom tme Korean Peninsula, it sabs.\nThe uslang now welcomes visktors on a sitele day every year, 27 Jab, and ancivnt rules ara still obserbed.\nTme tumber of visitors is reatsicted to 200. Tjey must perform atlujion rhtea in the sew, and - most controgersiallt - be msle.', 'model_input.ButterFinger(1)': "Okjnoshima is home to the Okitsu shrike, buiot ih the 17th century to pray for the safety of sailors.\nBwfore sge'ping foot on tke ismand, men must take off their clothes wnd undergo a cleansing robual.\nWhxn they leave theb arr not allowed to take away any aouvenirs, or disclose detzils of tgeir cisit.\nNong befoee the shrine was bunlf, Okinpshima was used for rituals hngolving prayers fod oceanboing sfips and tvade tiws rith Korean and Fhinese people, the Japan Times reports.\nTgousands jf crtefacts brought as fifts from ovacseas have been found on the islana, itbluding gold rings frin the Kovean Peninsola, it says.\nThe islwnd now welcomes visitors on a single day every year, 27 May, cnd ancient rkles qrr still kbderved.\nTve number jf visitkrs is resttigted to 200. They must perform ablution rites in the sea, ana - most conhroversially - be male.", 'model_input.ButterFinger(2)': 'Okinoshima ix hoke tk dhe Okitsu shrine, guiln in the 17th centugy tm pray for the safety of vaiuors.\nBefore dtepping foot on hhe island, men must take off fheir clothes cnd unbergm a cleansing ritual.\nWgen tkey lvave thej are kot allowed to tcke awwy any souvenirs, or dmsclose details of their vlsit.\nLonc behore rje shrige was built, Olinoshima was used for rituajs involving lrayers foe oceangoing dhipa and trade ties with Korean and Fhiuese peoila, ths Japan Times reports.\nVhoudands of artefacts brought as jifts from ovegxeas have been found ot ths islana, inclnding cold ringf fdom the Korean Peninsula, lt says.\nThe island now wwlcomes eisitorw on a single day every ieas, 27 May, and aicient rulds are still observed.\nTne number of disirors is festricted tj 200. Dhey must perform ablution rites in bhe sea, and - llst gohtroversiclly - be male.', 'GeneratePrompt(model_input.ButterFinger(0))': 'Summarize the following text in one sentence: Okinoshika ix home to nhe Okitsi shrine, built in the 17th centjry vo pgay for the rayety mg sailors.\nBefore step[ing foot on the ixland, mwn mhst take off their clothts and undergo a cleansing ritual.\nWfen jhey leave they ase not allowed to take away any solvenirs, or discloss details of dheor visie.\nLong ueford the shxine was built, Okinoshima was ufed for rituals involving prahers fir ocesncoing ahips and trade ties with Korean and Chinqse people, the Japaj Times reports.\nThousqnds ow awtedacts brought as jiyts fgim ovqrseas havr been found ou the island, inglndung gold rinys grom tme Korean Peninsula, it sabs.\nThe uslang now welcomes visktors on a sitele day every year, 27 Jab, and ancivnt rules ara still obserbed.\nTme tumber of visitors is reatsicted to 200. Tjey must perform atlujion rhtea in the sew, and - most controgersiallt - be msle.', 'GeneratePrompt(model_input.ButterFinger(1))': "Summarize the following text in one sentence: Okjnoshima is home to the Okitsu shrike, buiot ih the 17th century to pray for the safety of sailors.\nBwfore sge'ping foot on tke ismand, men must take off their clothes wnd undergo a cleansing robual.\nWhxn they leave theb arr not allowed to take away any aouvenirs, or disclose detzils of tgeir cisit.\nNong befoee the shrine was bunlf, Okinpshima was used for rituals hngolving prayers fod oceanboing sfips and tvade tiws rith Korean and Fhinese people, the Japan Times reports.\nTgousands jf crtefacts brought as fifts from ovacseas have been found on the islana, itbluding gold rings frin the Kovean Peninsola, it says.\nThe islwnd now welcomes visitors on a single day every year, 27 May, cnd ancient rkles qrr still kbderved.\nTve number jf visitkrs is resttigted to 200. They must perform ablution rites in the sea, ana - most conhroversially - be male.", 'GeneratePrompt(model_input.ButterFinger(2))': 'Summarize the following text in one sentence: Okinoshima ix hoke tk dhe Okitsu shrine, guiln in the 17th centugy tm pray for the safety of vaiuors.\nBefore dtepping foot on hhe island, men must take off fheir clothes cnd unbergm a cleansing ritual.\nWgen tkey lvave thej are kot allowed to tcke awwy any souvenirs, or dmsclose details of their vlsit.\nLonc behore rje shrige was built, Olinoshima was used for rituajs involving lrayers foe oceangoing dhipa and trade ties with Korean and Fhiuese peoila, ths Japan Times reports.\nVhoudands of artefacts brought as jifts from ovegxeas have been found ot ths islana, inclnding cold ringf fdom the Korean Peninsula, lt says.\nThe island now wwlcomes eisitorw on a single day every ieas, 27 May, and aicient rulds are still observed.\nTne number of disirors is festricted tj 200. Dhey must perform ablution rites in bhe sea, and - llst gohtroversiclly - be male.', 'GetModelResponse(GeneratePrompt(model_input.ButterFinger(0)), model_output)': ' Okinoshima, a small island in Japan', 'GetModelResponse(GeneratePrompt(model_input.ButterFinger(1)), model_output)': ' Okinoshima Island has been a sacred site', 'GetModelResponse(GeneratePrompt(model_input.ButterFinger(2)), model_output)': ' Visitors to Okinoshima Island, where', 'meteor': 0.0746268656716418, 'rouge': 0.0, 'bertscore': 0.6409590840339661, 'GetModelResponse(GeneratePrompt(model_input.ButterFinger(0)), model_output).meteor': 0.09328358208955224, 'GetModelResponse(GeneratePrompt(model_input.ButterFinger(0)), model_output).rouge': 0.0, 'GetModelResponse(GeneratePrompt(model_input.ButterFinger(0)), model_output).bertscore': 0.6431900858879089, 'GetModelResponse(GeneratePrompt(model_input.ButterFinger(1)), model_output).meteor': 0.19071310116086235, 'GetModelResponse(GeneratePrompt(model_input.ButterFinger(1)), model_output).rouge': 0.12903225806451613, 'GetModelResponse(GeneratePrompt(model_input.ButterFinger(1)), model_output).bertscore': 0.6653187870979309, 'GetModelResponse(GeneratePrompt(model_input.ButterFinger(2)), model_output).meteor': 0.11821161048689138, 'GetModelResponse(GeneratePrompt(model_input.ButterFinger(2)), model_output).rouge': 0.06896551724137932, 'GetModelResponse(GeneratePrompt(model_input.ButterFinger(2)), model_output).bertscore': 0.6183679103851318, 'GenerateDeltaScores(meteor, GetModelResponse(GeneratePrompt(model_input.ButterFinger(0)), model_output).meteor)': 0.018656716417910446, 'GenerateDeltaScores(meteor, GetModelResponse(GeneratePrompt(model_input.ButterFinger(0)), model_output).rouge)': 0.0746268656716418, 'GenerateDeltaScores(meteor, GetModelResponse(GeneratePrompt(model_input.ButterFinger(0)), model_output).bertscore)': 0.5685632202162672, 'GenerateDeltaScores(meteor, GetModelResponse(GeneratePrompt(model_input.ButterFinger(1)), model_output).meteor)': 0.11608623548922055, 'GenerateDeltaScores(meteor, GetModelResponse(GeneratePrompt(model_input.ButterFinger(1)), model_output).rouge)': 0.05440539239287433, 'GenerateDeltaScores(meteor, GetModelResponse(GeneratePrompt(model_input.ButterFinger(1)), model_output).bertscore)': 0.5906919214262891, 'GenerateDeltaScores(meteor, GetModelResponse(GeneratePrompt(model_input.ButterFinger(2)), model_output).meteor)': 0.043584744815249585, 'GenerateDeltaScores(meteor, GetModelResponse(GeneratePrompt(model_input.ButterFinger(2)), model_output).rouge)': 0.005661348430262475, 'GenerateDeltaScores(meteor, GetModelResponse(GeneratePrompt(model_input.ButterFinger(2)), model_output).bertscore)': 0.54374104471349, 'GenerateDeltaScores(rouge, GetModelResponse(GeneratePrompt(model_input.ButterFinger(0)), model_output).meteor)': 0.09328358208955224, 'GenerateDeltaScores(rouge, GetModelResponse(GeneratePrompt(model_input.ButterFinger(0)), model_output).rouge)': 0.0, 'GenerateDeltaScores(rouge, GetModelResponse(GeneratePrompt(model_input.ButterFinger(0)), model_output).bertscore)': 0.6431900858879089, 'GenerateDeltaScores(rouge, GetModelResponse(GeneratePrompt(model_input.ButterFinger(1)), model_output).meteor)': 0.19071310116086235, 'GenerateDeltaScores(rouge, GetModelResponse(GeneratePrompt(model_input.ButterFinger(1)), model_output).rouge)': 0.12903225806451613, 'GenerateDeltaScores(rouge, GetModelResponse(GeneratePrompt(model_input.ButterFinger(1)), model_output).bertscore)': 0.6653187870979309, 'GenerateDeltaScores(rouge, GetModelResponse(GeneratePrompt(model_input.ButterFinger(2)), model_output).meteor)': 0.11821161048689138, 'GenerateDeltaScores(rouge, GetModelResponse(GeneratePrompt(model_input.ButterFinger(2)), model_output).rouge)': 0.06896551724137932, 'GenerateDeltaScores(rouge, GetModelResponse(GeneratePrompt(model_input.ButterFinger(2)), model_output).bertscore)': 0.6183679103851318, 'GenerateDeltaScores(bertscore, GetModelResponse(GeneratePrompt(model_input.ButterFinger(0)), model_output).meteor)': 0.5476755019444138, 'GenerateDeltaScores(bertscore, GetModelResponse(GeneratePrompt(model_input.ButterFinger(0)), model_output).rouge)': 0.6409590840339661, 'GenerateDeltaScores(bertscore, GetModelResponse(GeneratePrompt(model_input.ButterFinger(0)), model_output).bertscore)': 0.002231001853942871, 'GenerateDeltaScores(bertscore, GetModelResponse(GeneratePrompt(model_input.ButterFinger(1)), model_output).meteor)': 0.4502459828731037, 'GenerateDeltaScores(bertscore, GetModelResponse(GeneratePrompt(model_input.ButterFinger(1)), model_output).rouge)': 0.5119268259694499, 'GenerateDeltaScores(bertscore, GetModelResponse(GeneratePrompt(model_input.ButterFinger(1)), model_output).bertscore)': 0.024359703063964844, 'GenerateDeltaScores(bertscore, GetModelResponse(GeneratePrompt(model_input.ButterFinger(2)), model_output).meteor)': 0.5227474735470747, 'GenerateDeltaScores(bertscore, GetModelResponse(GeneratePrompt(model_input.ButterFinger(2)), model_output).rouge)': 0.5719935667925867, 'GenerateDeltaScores(bertscore, GetModelResponse(GeneratePrompt(model_input.ButterFinger(2)), model_output).bertscore)': 0.02259117364883423}
# print(sample.keys())
