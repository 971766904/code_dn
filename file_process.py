# @Time : 2023/4/4 12:58 
# @Author : zhongyu 
# @File : file_process.py

from jddb.processor.basic_processors import ResamplingProcessor, NormalizationProcessor, ClipProcessor, TrimProcessor
from jddb.file_repo import FileRepo
from jddb.processor import ShotSet
from basic_processor import *
from ne_qa_processor import *


def find_tags(prefix, all_tags):
    """
        find tags that start with the prefix
    param:
        prefix: The first few strings of the tags users need to look for
        all_tags: a list of all the tags that needed to be filtered
    :return: matching tags as a list[sting]
    """
    return list(filter(lambda tag: tag.encode("utf-8").decode("utf-8", "ignore")[0:len(prefix)] == prefix, all_tags))


if __name__ == '__main__':
    change_file_repo = FileRepo("..//file_repo//ChangeShots//$shot_2$XX//$shot_1$X//")
    process_file_repo = FileRepo("..//file_repo//ProcessShots//$shot_2$XX//$shot_1$X//")
    target_tags = ['IP', 'BT', 'DV', 'DH', 'DENSITY', 'MP04', 'MP05', 'MP12', 'MP13', 'NP03', 'NP04', 'BOLU01',
                   'BOLU02', 'BOLU03', 'BOLU04', 'BOLU05', 'BOLU06', 'BOLU07', 'BOLU08', 'BOLU09', 'BOLU10', 'BOLU11',
                   'BOLU12', 'BOLU13', 'BOLU14', 'BOLU15', 'BOLU16', 'SX01', 'SX02',
                   'SX03', 'SX04', 'SX05', 'SX06', 'SX07', 'SX08', 'SX09', 'SX10', 'SX11', 'SX12', 'SX13', 'SX14',
                   'SX15', 'SX16', 'SX17', 'SX18', 'SX19', 'SX20']
    source_shotset = ShotSet(change_file_repo)
    # get all shots and tags from the file repo
    shot_list = source_shotset.shot_list
    valid_shots = []
    for shot in shot_list:
        all_tags = list(source_shotset.get_shot(shot).tags)
        if all(tag in all_tags for tag in target_tags):
            valid_shots.append(shot)
    valid_shotset = ShotSet(change_file_repo, valid_shots)

    # %%
    # 1. choose target tags
    processed_shotset = valid_shotset.remove_signal(tags=target_tags, keep=True,
                                                    save_repo=process_file_repo)

    # %%
    # 2. resample tags
    mirnov_tags = find_tags('MP', target_tags)
    array_tags = find_tags('SX', target_tags) + find_tags('BOLU', target_tags)
    processed_shotset = processed_shotset.process(processor=ResamplingProcessor(50000),
                                                  input_tags=mirnov_tags,
                                                  output_tags=mirnov_tags,
                                                  save_repo=process_file_repo)
    processed_shotset = processed_shotset.process(processor=ResamplingProcessor(1000),
                                                  input_tags=array_tags,
                                                  output_tags=array_tags,
                                                  save_repo=process_file_repo)
    processed_shotset = processed_shotset.process(processor=ResamplingProcessor(10000),
                                                  input_tags=['DENSITY', 'SX10'],
                                                  output_tags=['ne', 'sxr_core'],
                                                  save_repo=process_file_repo)
    processed_shotset = processed_shotset.process(processor=ResamplingProcessor(1000),
                                                  input_tags=['IP', 'BT', 'DV', 'DH'],
                                                  output_tags=['ip', 'bt', 'dz', 'dr'],
                                                  save_repo=process_file_repo)
    # drop useless raw signals
    processed_shotset = processed_shotset.remove_signal(tags=['IP', 'BT', 'DV', 'DH'],
                                                        save_repo=process_file_repo)

    # %%
    # 3.clip ,remove signal out side of the time of interests
    part_clip_tags = ['ip', 'bt', 'dz', 'dr', 'DENSITY']
    processed_shotset = processed_shotset.process(processor=ClipProcessor(start_time=0.05, end_time_label="DownTime"),
                                                  input_tags=part_clip_tags,
                                                  output_tags=part_clip_tags,
                                                  save_repo=process_file_repo)
    # %%
    # 4. trim part signal
    processed_shotset = processed_shotset.process(TrimProcessor(),
                                                  input_tags=[part_clip_tags],
                                                  output_tags=[part_clip_tags],
                                                  save_repo=process_file_repo)

    # %%
    # 5.qa & ne/nG
    processed_shotset = processed_shotset.process(processor=NormalizedDensity(a=0.4),
                                                  input_tags=[['DENSITY', 'ip']],
                                                  output_tags=['ne_nG'], save_repo=process_file_repo)
    processed_shotset = processed_shotset.process(processor=LimiterSecurityFactor(a=0.4, R=1.65),
                                                  input_tags=[['bt', 'ip']],
                                                  output_tags=['qa'], save_repo=process_file_repo)
