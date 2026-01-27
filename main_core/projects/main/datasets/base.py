"""
Base classes for Main datasets with common functionality.
"""
from functools import partial
from typing import Literal, Optional, Dict, List, Any
import torch
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from torch.utils.data import Dataset
from mmengine import print_log
from xtuner.registry import BUILDER
from .data_utils import dynamic_preprocess, template_map_fn, tokenize_conversation


class MainDatasetMixin:
    """
    Mixin class containing common functionality for Main datasets.
    This includes architecture configuration, image processing, and tokenization logic.
    """
    
    # Default constants
    DEFAULT_IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    DEFAULT_IMG_START_TOKEN = '<img>'
    DEFAULT_IMG_END_TOKEN = '</img>'
    
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    def _init_architecture_config(self, arch_type: Literal['intern_vl', 'upstreamllm', 'llava'] = 'intern_vl'):
        """Initialize architecture-specific configurations."""
        self.arch_type = arch_type
        
        # Set default tokens
        self.IMG_CONTEXT_TOKEN = self.DEFAULT_IMG_CONTEXT_TOKEN
        self.IMG_START_TOKEN = self.DEFAULT_IMG_START_TOKEN
        self.IMG_END_TOKEN = self.DEFAULT_IMG_END_TOKEN
        
        # Architecture-specific overrides
        if self.arch_type == 'upstreamllm':
            self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
            self.IMG_START_TOKEN = '<|vision_start|>'
            self.IMG_END_TOKEN = '<|vision_end|>'
        elif self.arch_type == 'llava':
            self.IMG_CONTEXT_TOKEN = '<image>'
            self.IMG_START_TOKEN = ''
            self.IMG_END_TOKEN = ''
    
    def _init_image_processing_config(self, 
                                    min_dynamic_patch: int = 1,
                                    max_dynamic_patch: int = 12,
                                    image_size: int = 448,
                                    use_thumbnail: bool = True,
                                    downsample_ratio: float = 0.5,
                                    patch_size: int = 14):
        """Initialize image processing configurations."""
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.use_thumbnail = use_thumbnail
        
        # Architecture-specific adjustments
        if self.arch_type == 'llava':
            self.downsample_ratio = 1
            self.image_size = 336
        else:
            self.downsample_ratio = downsample_ratio
            self.image_size = image_size
            
        # Calculate patch tokens
        if self.arch_type == 'upstreamllm':
            self.patch_token = 1
            self.min_pixels_single = 512*28*28
            self.max_pixels_single = 2048*28*28

            self.min_pixels_multi = 128*28*28
            self.max_pixels_multi = 512*28*28
        else:
            self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))
    
    def _init_tokenizer(self, tokenizer_config, special_tokens: Optional[List[str]] = None):
        """Initialize tokenizer with special tokens."""
        self.tokenizer = BUILDER.build(tokenizer_config)
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)
    
    def _init_image_processor(self, preprocessor_config=None):
        """Initialize image processor/transformer."""
        if preprocessor_config is None:
            self.transformer = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
            ])
            self.preprocessor = None
        else:
            self.transformer = None
            self.preprocessor = BUILDER.build(preprocessor_config)
    
    def _init_extra_image_processor(self, extra_image_processor_config=None):
        """Initialize extra image processor for grounding."""
        if extra_image_processor_config is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor_config)
        else:
            self.extra_image_processor = None
    
    def _setup_system_prompt(self):
        """Setup system prompt (empty by default for all architectures)."""
        self._system = ''
    
    def _process_single_image(self, image: Image.Image, single_image_mode: bool = False) -> Dict[str, Any]:
        """
        Process a single image and return pixel values and number of tokens.
        
        Args:
            image: PIL Image
            single_image_mode: Whether to use single image mode
            
        Returns:
            Dictionary containing processed image data
        """
        result = {}
        
        # Process for grounding if needed
        if hasattr(self, 'extra_image_processor') and self.extra_image_processor is not None:
            g_image = np.array(image)
            g_image = self.extra_image_processor.apply_image(g_image)
            g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
            result['g_pixel_values'] = g_pixel_values
        
        # Process images
        if self.preprocessor is not None:
            if self.arch_type == 'upstreamllm':
                images = [image]
                merge_length = self.preprocessor.image_processor.merge_size ** 2
                _data_dict = self.preprocessor.image_processor(
                    images=images, min_pixels=self.min_pixels_single, max_pixels=self.max_pixels_single
                )
                # _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                # _data_dict['image_grid_thw'] = torch.tensor(_data_dict['image_grid_thw'], dtype=torch.int)
                num_image_tokens = int(_data_dict['image_grid_thw'][0].prod()) // merge_length
            elif self.arch_type == 'llava':
                raise NotImplementedError("LLaVA preprocessor not implemented for single image mode")
                _data_dict = self.preprocessor(images, do_resize=True, size=(self.image_size, self.image_size))
                _data_dict['pixel_values'] = np.stack(_data_dict['pixel_values'], axis=0)
                _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                num_image_tokens = _data_dict['pixel_values'].shape[0] * self.patch_token
            else:
                raise NotImplementedError(f"Preprocessor not implemented for {self.arch_type}")
            result.update(_data_dict)
        else:
            assert self.transformer is not None, "Transformer must be defined if no preprocessor"
            # Prepare images for processing
            if single_image_mode:
                images = [image]
            else:
                images = dynamic_preprocess(image, self.min_dynamic_patch,
                                        self.max_dynamic_patch,
                                        self.image_size, self.use_thumbnail)

            pixel_values = [self.transformer(img) for img in images]
            pixel_values = torch.stack(pixel_values)
            result['pixel_values'] = pixel_values
            num_image_tokens = pixel_values.shape[0] * self.patch_token
        
        result['num_image_tokens'] = num_image_tokens
        return result
    
    def _process_multiple_images(self, images: List[Image.Image]) -> Dict[str, Any]:
        """
        Process multiple images (for video datasets) and return pixel values and number of tokens.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Dictionary containing processed image data
        """
        result = {}
        pixel_values = []
        extra_pixel_values = []
        
        # Process each image
        for image in images:
            image = image.convert('RGB')
            ori_width, ori_height = image.size
            
            # Process for grounding if needed
            if hasattr(self, 'extra_image_processor') and self.extra_image_processor is not None:
                g_image = np.array(image)
                g_image = self.extra_image_processor.apply_image(g_image)
                g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                extra_pixel_values.append(g_pixel_values)

            if self.preprocessor is not None:
                # Store images for batch processing
                pixel_values.append(image)
            else:
                # Apply transforms immediately
                transformed = self.transformer(image)
                pixel_values.append(transformed)

        # Process images based on preprocessor availability
        if self.preprocessor is not None:
            if self.arch_type == 'upstreamllm':
                merge_length = self.preprocessor.image_processor.merge_size ** 2
                _data_dict = self.preprocessor.image_processor(
                    images=images, min_pixels=self.min_pixels_multi, max_pixels=self.max_pixels_multi
                )
                num_frame_tokens = int(_data_dict['image_grid_thw'][0].prod() // merge_length)
                num_frames = _data_dict['image_grid_thw'].shape[0]
                num_total_tokens = num_frame_tokens * num_frames
                result.update(_data_dict)
                result['num_frame_tokens'] = num_frame_tokens
                result['num_frames'] = num_frames
            elif self.arch_type == 'llava':
                raise NotImplementedError("LLaVA preprocessor not implemented for multiple image mode")
            else:
                raise NotImplementedError(f"Preprocessor not implemented for {self.arch_type}")
        else:
            pixel_values = torch.stack(pixel_values, dim=0)  # (n_f, 3, h, w)
            result['pixel_values'] = pixel_values
            num_total_tokens = len(images) * self.patch_token

        if extra_pixel_values:
            result['g_pixel_values'] = extra_pixel_values

        result['num_image_tokens'] = num_total_tokens
        return result
    
    def _create_token_string(self, num_tokens: int, num_frames: int = 1) -> str:
        """
        Create token string for images or videos.
        
        Args:
            num_tokens: Total number of tokens
            num_frames: Number of frames (1 for image, >1 for video)
            
        Returns:
            Token string with proper formatting
        """
        if num_frames == 1:
            # Single image case
            return f'{self.IMG_START_TOKEN}{self.IMG_CONTEXT_TOKEN * num_tokens}{self.IMG_END_TOKEN}'
        else:
            # Video case - create frame tokens
            if self.arch_type == 'upstreamllm' and hasattr(self, 'patch_token') and self.patch_token == 1:
                # For upstreamllm with patch_token=1, we use single tokens that will be expanded later
                frame_token_str = f'{self.IMG_START_TOKEN}{self.IMG_CONTEXT_TOKEN}{self.IMG_END_TOKEN}'
            else:
                # For other cases, use tokens per frame
                tokens_per_frame = num_tokens // num_frames
                frame_token_str = f'{self.IMG_START_TOKEN}{self.IMG_CONTEXT_TOKEN * tokens_per_frame}{self.IMG_END_TOKEN}'
            
            # Repeat for all frames with newlines
            frame_tokens = (frame_token_str + '\n') * num_frames
            return frame_tokens.strip()
    
    def _create_image_token_string(self, num_image_tokens: int) -> str:
        """Create image token string for given number of tokens (backward compatibility)."""
        return self._create_token_string(num_image_tokens, num_frames=1)
    
    def _process_conversations_for_encoding(self, conversations: List[Dict], image_token_str: Optional[str] = None, 
                                          is_video: bool = False) -> List[Dict]:
        """
        Process conversations to prepare for tokenization.
        
        Args:
            conversations: List of conversation messages
            image_token_str: Image token string to replace <image> placeholders
            is_video: Whether this is video data (affects token placement)
            
        Returns:
            List of processed conversation turns
        """
        # Handle different input formats
        if conversations and 'input' in conversations[0] and 'output' in conversations[0]:
            # Already in the correct format (from video datasets)
            return conversations
            
        input_text = ''
        out_conversation = []
        
        # Skip leading GPT messages
        while conversations and conversations[0]['from'] == 'gpt':
            conversations = conversations[1:]
        
        conv_idx = 0
        for msg in conversations:
            if msg['from'] == 'human':
                value = msg['value']
                
                # Handle image token replacement
                if '<image>' in value:
                    if image_token_str is None:
                        value = value.replace('<image>', '')
                    else:
                        assert conv_idx == 0, f"Expected conversation index to be 0, but got {conv_idx} / {value}"
                        if is_video:
                            # For video, add tokens at the beginning
                            value = value.replace('<image>', '')
                            if conv_idx == 0:
                                value = image_token_str + value
                        else:
                            # For image, replace <image> placeholder
                            value = value.replace('<image>', image_token_str)
                        value = value.strip()
                
                input_text += value
            elif msg['from'] == 'gpt':
                out_conversation.append({
                    'input': input_text,
                    'output': msg['value'].strip()
                })
                input_text = ''
            else:
                raise NotImplementedError(f"Unknown message role: {msg['from']}")
            
            conv_idx += 1
        
        return out_conversation
    
    def get_inputid_labels(self, conversations: List[Dict]) -> Dict[str, List]:
        """
        Convert conversations to input_ids and labels for training.
        Uses video_lisa_encode_fn logic with template_map_fn support.
        
        Args:
            conversations: List of conversation messages (from/value or input/output format)
            image_token_str: Image token string to replace <image> placeholders
            
        Returns:
            Dictionary with 'input_ids' and 'labels' keys
        """
        # Prepare data dict for template_map_fn
        data_dict = {'conversation': conversations}
        result = self.template_map_fn(data_dict)
        data_dict.update(result)
        result = tokenize_conversation(data_dict, tokenizer=self.tokenizer, max_length=self.max_length)
        return result
    
    def _expand_video_tokens(self, conversations: List[Dict], num_frame_tokens: int, num_total_tokens: int) -> List[Dict]:
        """
        Expand video tokens for architectures that need post-processing (like upstreamllm).
        
        Args:
            conversations: Processed conversations
            num_frame_tokens: Tokens per frame
            num_total_tokens: Total video tokens
            
        Returns:
            Updated conversations with expanded tokens
        """
        if conversations and self.arch_type == 'upstreamllm' and hasattr(self, 'patch_token') and self.patch_token == 1:
            # For upstreamllm, expand the single tokens to frame tokens
            input_str = conversations[0]['input']
            input_str = input_str.replace(self.IMG_CONTEXT_TOKEN, self.IMG_CONTEXT_TOKEN * num_frame_tokens)
            assert input_str.count(self.IMG_CONTEXT_TOKEN) == num_total_tokens, \
                f"Token count mismatch: expected {num_total_tokens}, got {input_str.count(self.IMG_CONTEXT_TOKEN)}"
            conversations[0]['input'] = input_str
        return conversations
    
    def _get_modality_length_default(self, length: int = 100) -> int:
        """Get default modality length."""
        return length

    def _read_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Centralized image reading method to avoid duplicate code.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object or None if reading fails
        """
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            print(f'Error reading image {image_path}: {e}', flush=True)
            print_log(f'Error reading image {image_path}: {e}', logger='current')
            return None
    
    def _check_image_exists(self, image_path: str) -> bool:
        """
        Check if image file exists and can be opened without actually loading it.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if image exists and can be opened, False otherwise
        """
        try:
            with Image.open(image_path) as img:
                # Just check if we can open the image, don't load into memory
                img.verify()
            return True
        except Exception:
            return False


class MainBaseDataset(Dataset, MainDatasetMixin):
    """
    Base dataset class for Main datasets.
    Provides common initialization and utility methods.
    """
    
    def __init__(self,
                 tokenizer,
                 prompt_template,
                 max_length: int = 2048,
                 special_tokens: Optional[List[str]] = None,
                 arch_type: Literal['intern_vl', 'upstreamllm', 'llava'] = 'intern_vl',
                 preprocessor=None,
                 extra_image_processor=None,
                 min_dynamic_patch: int = 1,
                 max_dynamic_patch: int = 12,
                 image_size: int = 448,
                 use_thumbnail: bool = True,
                 downsample_ratio: float = 0.5,
                 patch_size: int = 14,
                 max_refetch: int = 1000,
                 repeats: float = 1.0,
                 name: str = "MainBaseDataset",
                 # distillation (optional)
                 # Backward compatible:
                 #   - distill_cache_dir: str | None
                 #   - distill_cache_type: "global"
                 # New (multi-teacher):
                 #   - distill_cache_dir: dict[name -> cache_dir]
                 #   - distill_cache_type: dict[name -> ("global"|"tokens"|"both")] or a single string
                 distill_cache_dir=None,
                 distill_cache_type='global',
                 ):
        """
        Initialize base dataset with common configurations.
        
        Args:
            tokenizer: Tokenizer configuration
            prompt_template: Template for formatting prompts
            max_length: Maximum sequence length
            special_tokens: List of special tokens to add
            arch_type: Architecture type ('intern_vl', 'upstreamllm', 'llava')
            preprocessor: Image preprocessor configuration
            extra_image_processor: Extra image processor for grounding
            min_dynamic_patch: Minimum dynamic patches
            max_dynamic_patch: Maximum dynamic patches  
            image_size: Image size
            use_thumbnail: Whether to use thumbnail
            downsample_ratio: Downsample ratio
            patch_size: Patch size
            max_refetch: Maximum refetch attempts
            repeats: Number of times to repeat the dataset (can be fractional, e.g., 0.2)
            template_map_fn: Template mapping function for xtuner format conversion
        """
        super().__init__()
        
        # Store core configurations
        self.template = prompt_template
        self.max_length = max_length
        self._max_refetch = max_refetch
        self.repeats = repeats
        
        # Pre-compute index mapping for equal distribution when using fractional repeats
        self._index_mapping = None
        
        # Template mapping function for format conversion
        self.template_map_fn = partial(template_map_fn, template=self.template)

        # Set name, it is for logging purposes
        self.name = name

        # Distillation cache config (optional)
        self.distill_cache_dir = distill_cache_dir
        self.distill_cache_type = distill_cache_type

        # Initialize architecture and processing configs
        self._init_architecture_config(arch_type)
        self._init_image_processing_config(min_dynamic_patch, max_dynamic_patch, 
                                         image_size, use_thumbnail, downsample_ratio, patch_size)
        
        # Initialize processors
        self._init_tokenizer(tokenizer, special_tokens)
        self._init_image_processor(preprocessor)
        self._init_extra_image_processor(extra_image_processor)
        self._setup_system_prompt()

    def _distill_cache_key_from_full_path(self, full_path: str) -> str:
        """Stable cache key for a frame path (avoid giant filenames).

        Note: key is hash of the *string* you pass in. To avoid mismatches between
        relative/absolute paths across machines, we normalize to an absolute path.
        """
        import hashlib
        import os
        norm = os.path.abspath(full_path)
        h = hashlib.sha1(norm.encode('utf-8')).hexdigest()[:16]
        return h

    def _distill_cache_key_candidates_from_full_path(self, full_path: str) -> List[str]:
        """Return multiple possible cache keys to be backward compatible.

        Historically, caches may be generated by hashing either:
        - relative path string (e.g. './data/xxx.jpg')
        - absolute path string (e.g. '/path/to/.../xxx.jpg')

        We first try absolute/real paths (recommended), then fall back to raw string.
        """
        import hashlib
        import os

        def _k(p: str) -> str:
            return hashlib.sha1(p.encode('utf-8')).hexdigest()[:16]

        # NOTE:
        # Different cache generators may hash:
        # - absolute paths
        # - paths relative to the repo cwd (often `Main/`)
        # - paths with a leading "./"
        # Add common relative variants to maximize cache hit rate.
        abs_p = os.path.abspath(full_path)
        real_p = os.path.realpath(full_path)
        candidates = [abs_p, real_p, full_path]
        try:
            cwd = os.getcwd()
            rel = os.path.relpath(real_p, start=cwd)
            candidates.extend([rel, f'./{rel}'])
        except Exception:
            pass

        out: List[str] = []
        seen = set()
        for c in candidates:
            if c in seen:
                continue
            seen.add(c)
            out.append(_k(c))
        return out

    def _load_distill_global_from_frame_paths(self, frame_paths: List[str]) -> Optional[torch.Tensor]:
        """
        Load cached teacher global embeddings for a list of frame file paths.

        Expected cache format:
        - Each frame has one file: <distill_cache_dir>/<sha1_16>.pt
        - Each file is a torch Tensor shaped [D] (global embedding)
        """
        if self.distill_cache_dir is None:
            return None
        if self.distill_cache_type not in ('global', 'both'):
            raise NotImplementedError(f"Unsupported distill_cache_type={self.distill_cache_type}")

        import os

        # NOTE:
        # For video datasets, we often sample multiple frames per sample. In practice, the
        # distill cache may not cover 100% of frames. The previous behavior was "all-or-nothing"
        # (return None if ANY frame is missing), which makes KD silently inactive most of the time.
        #
        # New behavior (best-effort):
        # - If all frames are found: return Tensor[T, D] (frame-level teacher).
        # - If only a subset is found: return Tensor[D] (mean pooled teacher), which the model
        #   will broadcast/repeat to match student frames.
        # - If none is found: return None.
        feats = []
        n_missing = 0
        base_dir = self.distill_cache_dir.rstrip('/')
        for full_path in frame_paths:
            feat = None
            for key in self._distill_cache_key_candidates_from_full_path(full_path):
                cache_file = f"{base_dir}/{key}.pt"
                if not os.path.exists(cache_file):
                    continue
                try:
                    payload = torch.load(cache_file, map_location='cpu')
                    # Backward compatible:
                    # - old format: Tensor[D]
                    # - new format: {"global": Tensor[D], "tokens": Tensor[N,D], ...}
                    if torch.is_tensor(payload):
                        feat = payload
                    elif isinstance(payload, dict) and "global" in payload:
                        feat = payload["global"]
                    else:
                        feat = None
                except Exception:
                    feat = None
                    continue
                break
            if feat is None or (not torch.is_tensor(feat)) or feat.ndim != 1:
                n_missing += 1
                continue
            feats.append(feat)

        if len(feats) == 0:
            return None
        if n_missing == 0 and len(feats) == len(frame_paths):
            # full coverage -> frame-level teacher
            return torch.stack(feats, dim=0)
        # partial coverage -> clip-level teacher
        return torch.stack(feats, dim=0).mean(dim=0)

    def _load_distill_tokens_from_frame_paths(self, frame_paths: List[str]) -> Optional[torch.Tensor]:
        """
        Load cached teacher token embeddings for a list of frame file paths.

        Expected cache format:
        - Each frame has one file: <distill_cache_dir>/<sha1_16>.pt
        - File can be:
            - Tensor[N, D] (tokens only)
            - {"tokens": Tensor[N, D], ...} (multi-field cache)
        """
        if self.distill_cache_dir is None:
            return None
        if self.distill_cache_type not in ('tokens', 'both'):
            raise NotImplementedError(f"Unsupported distill_cache_type={self.distill_cache_type}")

        import os

        feats = []
        base_dir = self.distill_cache_dir.rstrip('/')
        for full_path in frame_paths:
            feat = None
            for key in self._distill_cache_key_candidates_from_full_path(full_path):
                cache_file = f"{base_dir}/{key}.pt"
                if not os.path.exists(cache_file):
                    continue
                try:
                    payload = torch.load(cache_file, map_location='cpu')
                    if torch.is_tensor(payload):
                        feat = payload
                    elif isinstance(payload, dict) and "tokens" in payload:
                        feat = payload["tokens"]
                    else:
                        feat = None
                except Exception:
                    feat = None
                    continue
                break
            if feat is None or (not torch.is_tensor(feat)) or feat.ndim != 2:
                return None
            feats.append(feat)
        # [T, N, D] (note: N may vary across frames/teachers; stack may fail)
        try:
            return torch.stack(feats, dim=0)
        except Exception:
            return None

    def _load_distill_from_frame_paths_multi(self, frame_paths: List[str]):
        """
        Multi-teacher distillation loader.

        Returns:
          (global_map, tokens_map, valid_map)
        where each map is teacher_name -> Tensor/None.
        """
        if self.distill_cache_dir is None:
            return {}, {}, {}

        # Single-teacher (backward compatible)
        if not isinstance(self.distill_cache_dir, dict):
            g = None
            t = None
            if self.distill_cache_type in ('global', 'both'):
                g = self._load_distill_global_from_frame_paths(frame_paths)
            if self.distill_cache_type in ('tokens', 'both'):
                t = self._load_distill_tokens_from_frame_paths(frame_paths)
            ok = (g is not None) or (t is not None)
            return ({'default': g} if g is not None else {}), ({'default': t} if t is not None else {}), {'default': ok}

        # Multi-teacher
        global_map = {}
        tokens_map = {}
        valid_map = {}
        type_map = self.distill_cache_type if isinstance(self.distill_cache_type, dict) else {}
        for name, d in self.distill_cache_dir.items():
            if not d:
                continue

            # Reuse single-loader helpers by temporarily swapping dir/type
            old_dir = self.distill_cache_dir
            old_type = self.distill_cache_type
            self.distill_cache_dir = d
            ttype = type_map.get(name, old_type if isinstance(old_type, str) else 'global')
            self.distill_cache_type = ttype
            g = None
            t = None
            try:
                if ttype in ('global', 'both'):
                    g = self._load_distill_global_from_frame_paths(frame_paths)
                if ttype in ('tokens', 'both'):
                    t = self._load_distill_tokens_from_frame_paths(frame_paths)
            finally:
                self.distill_cache_dir = old_dir
                self.distill_cache_type = old_type

            if g is not None:
                global_map[name] = g
            if t is not None:
                tokens_map[name] = t
            valid_map[name] = (g is not None) or (t is not None)
        return global_map, tokens_map, valid_map
    
    def __len__(self):
        """Get total length considering repeats."""
        return int(self.real_len() * self.repeats)
    
    def real_len(self):
        """Get the actual length without repeats. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement real_len")
    
    def _get_index_mapping(self):
        """Create or return cached index mapping for shuffled samples with fractional repeats."""
        if self._index_mapping is None:
            real_length = self.real_len()
            total_length = int(real_length * self.repeats)
            
            # Create indices based on repeats
            if self.repeats >= 1.0:
                # For repeats >= 1, repeat indices and take the first total_length
                repeated_indice = np.tile(np.arange(real_length), int(np.ceil(self.repeats)))
                indices = np.random.permutation(repeated_indice)[:total_length]
            else:
                # For repeats < 1, randomly sample total_length indices from all available indices
                indices = np.random.choice(real_length, size=total_length, replace=False)
            
            self._index_mapping = indices
                
        return self._index_mapping
    
    def shuffle_indices(self):
        """Create a new shuffled index mapping. Call this after each epoch for different sample order."""
        # Reset the index mapping to None so it gets recreated with new shuffle
        self._index_mapping = None
        # Force creation of new index mapping
        self._get_index_mapping()
    
    def __getitem__(self, index):
        """Unified __getitem__ implementation with refetch logic."""
        # Handle repeats using index mapping for equal distribution
        index_mapping = self._get_index_mapping()
        mapped_index = index_mapping[index]
        
        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(mapped_index)
            # Broken images may cause the returned data to be None
            if data is None:
                mapped_index = self._rand_another_index()
                continue
            return data
        
        # If we reach here, all retries failed
        raise RuntimeError(f"Failed to get valid data after {self._max_refetch + 1} attempts")
    
    def _rand_another_index(self) -> int:
        """Get random index for refetching."""
        return np.random.randint(0, self.real_len())
    
    def prepare_data(self, index):
        """Prepare data for a given index. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement prepare_data")
    
    @property
    def modality_length(self):
        """Get modality length for all items."""
        return [self._get_modality_length_default() for _ in range(len(self))]
