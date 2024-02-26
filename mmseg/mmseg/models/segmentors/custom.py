from mmseg.registry import MODELS
from mmseg.models.segmentors import EncoderDecoder
from mmseg.structures.seg_data_sample import SegDataSample
from mmseg.models.utils.wrappers import resize

from mmengine.structures import PixelData


class PostProcessResultMixin:
    def postprocess_result(self, seg_logits, data_samples):
        """Convert results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                model of each input image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if "img_padding_size" not in img_meta:
                    padding_size = img_meta.get("padding_size", [0] * 4)
                else:
                    padding_size = img_meta["img_padding_size"]
                padding_left, padding_right, padding_top, padding_bottom = padding_size
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[
                    i : i + 1,
                    :,
                    padding_top : H - padding_bottom,
                    padding_left : W - padding_right,
                ]

                flip = img_meta.get("flip", None)
                if flip:
                    flip_direction = img_meta.get("flip_direction", None)
                    assert flip_direction in ["horizontal", "vertical"]
                    if flip_direction == "horizontal":
                        i_seg_logits = i_seg_logits.flip(dims=(3,))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2,))

                # resize as original shape
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta["ori_shape"],
                    mode="bilinear",
                    align_corners=self.align_corners,
                    warning=False,
                ).squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            i_seg_logits = i_seg_logits.sigmoid()
            i_seg_pred = (i_seg_logits > 0.5).to(i_seg_logits)

            data_samples[i].set_data(
                {
                    "seg_logits": PixelData(**{"data": i_seg_logits}),
                    "pred_sem_seg": PixelData(**{"data": i_seg_pred}),
                }
            )

        return data_samples


@MODELS.register_module()
class EncoderDecoderWithoutArgmax(PostProcessResultMixin, EncoderDecoder):
    pass