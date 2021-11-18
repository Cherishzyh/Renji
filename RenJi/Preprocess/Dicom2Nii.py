import os
import time
from pathlib import Path
import shutil

from MIP4AIM.Utility.DicomInfo import DicomShareInfo
from MIP4AIM.Dicom2Nii.DataReader import DataReader
from MIP4AIM.Dicom2Nii.Dicom2Nii import ConvertDicom2Nii
from MIP4AIM.Application2Series.ManufactureMatcher import SeriesCopyer
from MIP4AIM.NiiProcess.DwiProcessor import DwiProcessor
from MIP4AIM.NiiProcess.Registrator import Registrator

from MeDIT.Log import CustomerCheck, Eclog


class Dcm2Nii:
    def __init__(self, raw_folder, processed_folder, failed_folder, is_overwrite=False):
        self.raw_folder = raw_folder
        self.process_folder = processed_folder
        self.failed_folder = failed_folder
        self.is_overwrite = is_overwrite
        self.dcm2niix_path = r'D:\Project\dcm2niix_win\dcm2niix.exe'

        self.dicom_info = DicomShareInfo()
        self.data_reader = DataReader()

        self.series_copyer = SeriesCopyer()
        self.dwi_processor = DwiProcessor()
        self.registrator = Registrator()

    def GetPath(self, case_folder):
        for root, dirs, files in os.walk(case_folder):
            if len(files) != 0:
                yield root, dirs, files

    def RegistrateBySpacing(self, case_folder, target_b_value=1500):
        t2_path = os.path.join(case_folder, 't2.nii')
        adc_path = os.path.join(case_folder, 'adc.nii')
        dwi_path = self.dwi_processor.ExtractSpecificDwiFile(case_folder, target_b_value)

        if dwi_path == '':
            return False, 'No DWI with b close to {}'.format(target_b_value)

        self.registrator.fixed_image = t2_path

        self.registrator.moving_image = adc_path
        try:
            self.registrator.RegistrateBySpacing(store_path=self.registrator.GenerateStorePath(adc_path))
        except:
            return False, 'Align ADC Failed'

        self.registrator.moving_image = dwi_path
        try:
            self.registrator.RegistrateBySpacing(store_path=self.registrator.GenerateStorePath(dwi_path))
        except:
            return False, 'Align DWI Failed'

        return True, ''

    def SeperateDWI(self, case_folder):
        self.dwi_processor.Seperate4DDwiInCaseFolder(case_folder)

    def ConvertDicom2Nii(self, case_folder):
        for root, dirs, files in os.walk(case_folder, topdown=False):
            # it is possible to one series that storing the DICOM
            if len(files) > 3 and len(dirs) == 0:
                # if self.dicom_info.IsDICOMFolder(root):
                ConvertDicom2Nii(Path(root), Path(root), dcm2niix_path=self.dcm2niix_path)

    def MoveFilaedCase(self, case):
        if not os.path.exists(os.path.join(self.failed_folder, case)):
            shutil.move(os.path.join(self.raw_folder, case), os.path.join(self.failed_folder, case))
        else:
            add_time = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
            shutil.move(os.path.join(self.raw_folder, case), os.path.join(self.failed_folder, '_{}'.format(add_time)))
        if os.path.exists(os.path.join(self.process_folder, case)):
            shutil.rmtree(os.path.join(self.process_folder, case))

    def InerativeCase(self):
        self.log = CustomerCheck(os.path.join(self.failed_folder, 'failed_log.csv'), patient=1,
                                 data={'State': [], 'Info': []})
        self.eclog = Eclog(os.path.join(self.failed_folder, 'failed_log_details.log')).GetLogger()
        for case in os.listdir(self.raw_folder):
            case_folder = os.path.join(self.raw_folder, case)

            print('Convert Dicom to Nii:\n {}'.format(case_folder))
            try:
                self.ConvertDicom2Nii(case_folder)
            except Exception as e:
                self.log.AddOne(case_folder, {'State': 'Dicom to Nii failed.', 'Info': e.__str__()})
                self.eclog.error(e)
                # self.MoveFilaedCase(case_folder)
                continue

# def CopyNii(src_root, des_root):
#     if not os.path.exists(des_root):
#         os.mkdir(des_root)
#     for root, dirs, files in os.walk(src_root):
#         if len(files) > 0 and len(dirs) == 0:
#             case_name = Path(root).name
#             case_root = Path(root).parent
#             while True:
#                 if len(str(case_name)) > 9:
#                     break
#                 else:
#                     case_name = Path(case_root).name
#                     case_root = Path(case_root).parent
#             des_folder = os.path.join(des_root, case_name)
#             if not os.path.exists(des_folder):
#                 os.mkdir(des_folder)
#             [shutil.copyfile(os.path.join(root, file), os.path.join(des_folder, file)) for file in files if file.endswith('.nii')]

def CopyNii(src_root, des_root):
    if not os.path.exists(des_root):
        os.mkdir(des_root)
    for case in os.listdir(src_root):
        try:
            case_name = '{} {}'.format(case.split(' ')[0], case.split(' ')[1])
            src_folder = os.path.join(src_root, '{}/{}/2'.format(case, case))
            a = [file for file in os.listdir(src_folder) if file.endswith('.nii')]
            if len(a) == 0:
                print(case)
            [shutil.copyfile(os.path.join(src_folder, file), os.path.join(des_root, '{}.nii'.format(case_name))) for file in os.listdir(src_folder) if file.endswith('.nii')]
        except Exception as e:
            print(case, e)

if __name__ == '__main__':
    # raw_folder = r'Z:\RenJi\ExternalTest\TVT'
    # store_folder = r'Z:\RenJi\ExternalTest\CH2'
    # failed_folder = r'Z:\RenJi\ExternalTest\CH2'
    processor = Dcm2Nii(raw_folder=r'Z:\RenJi\ExternalTest\external test\20201130 liwenling 1',
                        processed_folder=r'Z:\RenJi\ExternalTest\external test\20201130 liwenling 1',
                        failed_folder=r'Z:\RenJi\ExternalTest\external test\20201130 liwenling 1', is_overwrite=True)
    # processor.InerativeCase()x
    processor.ConvertDicom2Nii(r'Z:\RenJi\ExternalTest\external test\20201130 liwenling 1\3')
    # for case in os.listdir(r'C:\Users\ZhangYihong\Desktop\aaaa\OriginalPath\CSJ^chen shi jie'):
    # processor.SeperateDWI(r'C:\Users\ZhangYihong\Desktop\aaaa\OriginalPath\XSJ^xu shou jun')
    # CopyNii(raw_folder, store_folder)
    # for case in os.listdir(store_folder):
    #     case_folder = os.path.join(store_folder, case)
    #     if len(os.listdir(case_folder)) == 0:
    #         print(case)

    # for root, dirs, files in os.walk(r'D:\Data\renji0722\RawData\3\3\20151121 normal 2'):
    #     if len(files) > 0 and len(dirs) == 0:
    #        ConvertDicom2Nii(root, root, dcm2niix_path=r'D:\Project\dcm2niix_win\dcm2niix.exe')
    # CopyNii(r'D:\Data\renji0722\RawData\3\3\20151121 normal 2',
    #         r'D:\Data\renji0722\ProcessData\3')
